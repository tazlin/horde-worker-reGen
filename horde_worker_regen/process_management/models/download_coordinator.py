"""Coordinate background model downloads and startup gates."""

from __future__ import annotations

import time
from collections.abc import Callable

from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from loguru import logger

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeDownloadAvailabilityMessage
from horde_worker_regen.process_management.ipc.supervisor_channel import DownloadPlanSummary
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.desired_state import DesiredState
from horde_worker_regen.process_management.models.model_availability import ModelAvailability
from horde_worker_regen.reporting.status_reporter import StatusReporter


class ModelDownloadCoordinator:
    """Coordinate model availability, download requests, and lazy process startup."""

    DOWNLOAD_STARTUP_GRACE_SECONDS = 90.0
    """Seconds to wait for the first availability report before starting worker processes anyway."""

    DOWNLOAD_PLAN_REFRESH_SECONDS = 2.0
    """Seconds to cache the disk-plan summary before recomputing it."""

    def __init__(
        self,
        *,
        state: WorkerState,
        process_map: ProcessMap,
        process_lifecycle: ProcessLifecycleManager,
        model_availability: ModelAvailability,
        desired_state: DesiredState,
        bridge_data_provider: Callable[[], reGenBridgeData],
        stable_diffusion_reference_provider: Callable[[], dict[str, ImageGenerationModelRecord] | None],
        enable_background_downloads: bool,
        clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the download coordinator.

        Args:
            state: Mutable worker state shared with other orchestration collaborators.
            process_map: Live process map used for process-count checks.
            process_lifecycle: Process lifecycle facade used to start processes and send download commands.
            model_availability: Authoritative model availability reported by the download process.
            desired_state: Operator/config desired-model state.
            bridge_data_provider: Return the current live bridge data.
            stable_diffusion_reference_provider: Return the loaded image-model reference, if available.
            enable_background_downloads: Whether the background download process is enabled.
            clock: Wall-clock provider for startup-grace decisions.
            monotonic_clock: Monotonic clock provider for download-plan cache expiry.
        """
        self._state = state
        self._process_map = process_map
        self._process_lifecycle = process_lifecycle
        self._model_availability = model_availability
        self._desired_state = desired_state
        self._bridge_data_provider = bridge_data_provider
        self._stable_diffusion_reference_provider = stable_diffusion_reference_provider
        self._enable_background_downloads = enable_background_downloads
        self._clock = clock
        self._monotonic_clock = monotonic_clock

        self.inference_processes_started = False
        self.safety_processes_started = False
        self.initial_download_requested = False
        self.download_wait_started = 0.0
        self.download_plan_summary: DownloadPlanSummary | None = None
        self.download_plan_refreshed_at = 0.0

    @property
    def bridge_data(self) -> reGenBridgeData:
        """Return the current live bridge data."""
        return self._bridge_data_provider()

    def on_download_availability(self, message: HordeDownloadAvailabilityMessage) -> None:
        """Record an on-disk availability snapshot from the download process."""
        self._model_availability.update(
            present=set(message.available_model_names),
            currently_downloading=message.currently_downloading,
            pending=tuple(message.pending_downloads),
            failed=tuple(message.failed_downloads),
            status=message.status,
            scan_complete=message.scan_complete,
            safety_present=message.safety_models_present,
            safety_attempted=message.safety_models_attempted,
            controlnet_present=message.controlnet_present,
            sdxl_controlnet_present=message.sdxl_controlnet_present,
            post_processing_present=message.post_processing_present,
            controlnet_failed=message.controlnet_failed,
        )

        if message.reference_changed:
            self._process_lifecycle.broadcast_reload_model_database()

        if message.scan_complete and not self.initial_download_requested:
            self.initial_download_requested = True
            plan = self.get_download_plan_summary()
            if plan is not None:
                StatusReporter.log_startup_download_plan(plan)
            self.reconcile_downloads(run_aux_if_incomplete=True)

        self.maybe_start_safety_processes()
        self.maybe_start_inference_processes()

    def reconcile_downloads(
        self,
        *,
        run_aux_if_incomplete: bool = False,
        force_aux: bool = False,
        previously_configured: set[str] | None = None,
    ) -> None:
        """Drive the download process toward the desired on-disk model set."""
        if not self._enable_background_downloads:
            return
        present = self._model_availability.present or set()
        in_flight = set(self._model_availability.pending)
        if self._model_availability.currently_downloading is not None:
            in_flight.add(self._model_availability.currently_downloading)
        plan = self._desired_state.reconcile(
            configured=self.bridge_data.image_models_to_load,
            present=present,
            in_flight=in_flight,
        )
        removed = (previously_configured or set()) - plan.desired
        download_aux = force_aux or (run_aux_if_incomplete and len(plan.to_fetch) > 0)
        if not plan.has_work and not removed and not download_aux:
            return
        if removed:
            logger.info(f"Config removed {len(removed)} image model(s); stopping their downloads: {sorted(removed)}")
        if plan.to_fetch:
            desired_present = len(plan.desired) - len(plan.to_fetch)
            logger.info(
                f"Worker has {desired_present} of {len(plan.desired)} desired models on disk; "
                f"background-downloading {len(plan.to_fetch)} missing: {list(plan.to_fetch)}",
            )
        self._process_lifecycle.request_downloads(
            list(plan.to_fetch),
            download_aux=download_aux,
            desired_image_models=sorted(plan.desired),
        )

    def download_process_flags(self) -> tuple[object, ...]:
        """Return download-gating bridge-data fields for change detection."""
        bridge_data = self.bridge_data
        return (
            bridge_data.nsfw,
            bridge_data.allow_lora,
            bridge_data.allow_controlnet,
            bridge_data.allow_sdxl_controlnet,
            bridge_data.allow_post_processing,
            bridge_data.purge_loras_on_download,
        )

    def forward_download_gating_if_changed(self, previous_flags: tuple[object, ...]) -> None:
        """Apply changed download-gating flags to the download process live."""
        if not self._enable_background_downloads:
            return
        if self.download_process_flags() == previous_flags:
            return
        bridge_data = self.bridge_data
        logger.info("Download-affecting config changed on reload; applying the new gating live.")
        self._process_lifecycle.set_download_gating(
            nsfw=bridge_data.nsfw,
            allow_lora=bridge_data.allow_lora,
            allow_controlnet=bridge_data.allow_controlnet,
            allow_sdxl_controlnet=bridge_data.allow_sdxl_controlnet,
            allow_post_processing=bridge_data.allow_post_processing,
            purge_loras=bridge_data.purge_loras_on_download,
        )

    def enter_downloads_only_hold(self) -> None:
        """Enter the download-only posture."""
        if not self._enable_background_downloads:
            logger.warning("Download-only hold requested but background downloads are disabled; ignoring.")
            return
        if self._state.downloads_only_hold:
            return
        self._state.downloads_only_hold = True
        self._process_lifecycle.start_download_process()
        logger.info("Entered download-only mode: pre-fetching models; inference and job popping are held.")

    def leave_downloads_only_hold(self) -> None:
        """Leave the download-only posture and bring the worker fully up."""
        if not self._state.downloads_only_hold:
            return
        self._state.downloads_only_hold = False
        logger.info("Leaving download-only mode (GO_LIVE): starting inference/safety and resuming job popping.")
        if self.inference_processes_started and self._process_map.num_inference_processes() == 0:
            self.inference_processes_started = False
        self.maybe_start_safety_processes()
        self.maybe_start_inference_processes()

    def download_models_on_demand(self, model_names: list[str], *, include_aux: bool) -> None:
        """Add operator-chosen models to the desired set and fetch them now."""
        if not self._enable_background_downloads:
            logger.warning("On-demand download requested but background downloads are disabled; ignoring.")
            return
        if not model_names and not include_aux:
            return
        self._process_lifecycle.start_download_process()
        if model_names:
            self._desired_state.add_picker_models(model_names)
            logger.info(f"Picker added {len(model_names)} model(s) to the desired set: {sorted(model_names)}")
        self.reconcile_downloads(force_aux=include_aux)

    def maybe_start_safety_processes(self) -> None:
        """Start safety processes once the required safety models are on disk."""
        if self.safety_processes_started or not self._enable_background_downloads:
            return
        if self._state.downloads_only_hold:
            return

        availability = self._model_availability
        if availability.safety_present:
            logger.info("Required safety models are present on disk; starting safety processes")
            self._process_lifecycle.start_safety_processes()
            self.safety_processes_started = True
            return

        if availability.safety_attempted:
            logger.warning(
                "Download process finished without providing the safety models; starting the safety "
                "process to fetch them directly (it will surface any download error)",
            )
            self._process_lifecycle.start_safety_processes()
            self.safety_processes_started = True
            return

        if not availability.is_known and (self._clock() - self.download_wait_started) > (
            self.DOWNLOAD_STARTUP_GRACE_SECONDS
        ):
            logger.warning(
                "No model availability report after "
                f"{self.DOWNLOAD_STARTUP_GRACE_SECONDS:.0f}s; starting safety processes anyway",
            )
            self._process_lifecycle.start_safety_processes()
            self.safety_processes_started = True

    def maybe_start_inference_processes(self) -> None:
        """Start inference processes once at least one model is present."""
        if self.inference_processes_started or not self._enable_background_downloads:
            return
        if self._state.downloads_only_hold:
            return

        availability = self._model_availability

        # An alchemist-only worker (no image models configured, e.g. a CPU install) will never see an
        # image model land, so it must not wait for one: start inference as soon as the on-disk scan has
        # completed, so the alchemy graph forms have their process. Without this the worker would wait
        # forever (none of the model-present branches below can fire with an empty configured set).
        if not self.bridge_data.image_models_to_load and self.bridge_data.alchemist and availability.scan_complete:
            logger.info("Alchemist-only worker (no image models configured); starting inference processes")
            self._process_lifecycle.start_inference_processes()
            self.inference_processes_started = True
            return

        if availability.scan_complete and len(availability.present or set()) > 0:
            logger.info("At least one model is present on disk; starting inference processes")
            self._process_lifecycle.start_inference_processes()
            self.inference_processes_started = True
            return

        if availability.is_known and not availability.scan_complete:
            self.download_wait_started = self._clock()
            return

        if not availability.is_known and (self._clock() - self.download_wait_started) > (
            self.DOWNLOAD_STARTUP_GRACE_SECONDS
        ):
            logger.warning(
                "No model availability report after "
                f"{self.DOWNLOAD_STARTUP_GRACE_SECONDS:.0f}s; starting inference processes anyway",
            )
            self._process_lifecycle.start_inference_processes()
            self.inference_processes_started = True

    def get_download_plan_summary(self) -> DownloadPlanSummary | None:
        """Compute the config's disk-implications summary on a short cache."""
        now = self._monotonic_clock()
        fresh = (now - self.download_plan_refreshed_at) < self.DOWNLOAD_PLAN_REFRESH_SECONDS
        if self.download_plan_summary is not None and fresh:
            return self.download_plan_summary

        reference = self._stable_diffusion_reference_provider()
        if reference is None:
            return self.download_plan_summary

        from horde_worker_regen import model_download_plan

        plan = model_download_plan.compute_download_plan(
            list(self.bridge_data.image_models_to_load),
            reference,
            extra_model_directories=self.bridge_data.extra_model_directories,
        )
        self.download_plan_summary = DownloadPlanSummary(
            present_bytes=plan.present_bytes,
            to_download_bytes=plan.to_download_bytes,
            total_bytes=plan.total_bytes,
            free_disk_bytes=plan.free_disk_bytes,
            fits=plan.fits,
            shortfall_bytes=plan.shortfall_bytes,
            num_present=plan.num_present,
            num_to_download=plan.num_to_download,
            sizes_complete=plan.sizes_complete,
        )
        self.download_plan_refreshed_at = now
        return self.download_plan_summary
