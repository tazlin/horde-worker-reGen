"""Contains the code to download all models specified in the config file. Executable as a standalone script."""


def download_all_models(
    *,
    load_config_from_env_vars: bool = False,
    purge_unused_loras: bool = False,
    directml: int | None = None,
) -> None:
    """Download all models specified in the config file."""
    from horde_worker_regen.load_env_vars import load_env_vars_from_config

    if not load_config_from_env_vars:
        load_env_vars_from_config()

    from loguru import logger

    from horde_worker_regen.bridge_data.load_config import BridgeDataLoader, reGenBridgeData
    from horde_worker_regen.consts import BRIDGE_CONFIG_FILENAME
    from horde_worker_regen.reference_helper import ensure_model_reference_manager_initialized

    horde_model_reference_manager = ensure_model_reference_manager_initialized()

    bridge_data: reGenBridgeData | None = None
    try:
        if not load_config_from_env_vars:
            bridge_data = BridgeDataLoader.load(
                file_path=BRIDGE_CONFIG_FILENAME,
                horde_model_reference_manager=horde_model_reference_manager,
            )
            bridge_data.load_env_vars()
            bridge_data.prepare_custom_models()
        else:
            bridge_data = BridgeDataLoader.load_from_env_vars(
                horde_model_reference_manager=horde_model_reference_manager,
            )
    except Exception as e:
        logger.error(e)
        input("Press any key to exit...")

    if bridge_data is None:
        logger.error("Failed to load bridge data")
        exit(1)

    import hordelib
    from horde_safety.deep_danbooru_model import download_deep_danbooru_model
    from horde_safety.interrogate import get_interrogator_no_blip

    download_deep_danbooru_model()
    _ = get_interrogator_no_blip()
    del _

    extra_comfyui_args = []
    if directml is not None:
        extra_comfyui_args.append(f"--directml={directml}")

    hordelib.initialise(extra_comfyui_args=extra_comfyui_args)
    from hordelib.api import SharedModelManager

    SharedModelManager.load_model_managers()

    if bridge_data.allow_lora:
        if SharedModelManager.manager.lora is None:
            logger.error("Failed to load LORA model manager")
            exit(1)
        SharedModelManager.manager.lora.reset_adhoc_cache()
        SharedModelManager.manager.lora.download_default_models(nsfw=bridge_data.nsfw)
        SharedModelManager.manager.lora.wait_for_downloads(600)
        SharedModelManager.manager.lora.wait_for_adhoc_reset(120)

    if purge_unused_loras or bridge_data.purge_loras_on_download:
        logger.info("Purging unused LORAs...")
        if SharedModelManager.manager.lora is None:
            logger.error("Failed to load LORA model manager")
            exit(1)
        deleted_loras = SharedModelManager.manager.lora.delete_unused_models(30)
        logger.success(f"Purged {len(deleted_loras)} unused LORAs.")

    if bridge_data.allow_controlnet:
        if SharedModelManager.manager.controlnet is None:
            logger.error("Failed to load controlnet model manager")
            exit(1)
        for cn_model in SharedModelManager.manager.controlnet.model_reference:
            if (
                cn_model not in SharedModelManager.manager.controlnet.available_models
                and "sdxl" in cn_model.lower()
                and not bridge_data.allow_sdxl_controlnet
            ):
                logger.warning(f"Skipping download of {cn_model} because `allow_sdxl_controlnet` is false.")
                continue

            SharedModelManager.manager.controlnet.download_model(cn_model)
            SharedModelManager.manager.controlnet.download_model(cn_model)
        if not SharedModelManager.preload_annotators():
            logger.error("Failed to download the controlnet annotators")
            exit(1)

    if bridge_data.allow_sdxl_controlnet:
        if SharedModelManager.manager.miscellaneous is None:
            logger.error("Failed to load miscellaneous model manager")
            exit(1)
        SharedModelManager.manager.miscellaneous.download_all_models()
        SharedModelManager.manager.miscellaneous.download_all_models()
        for model in SharedModelManager.manager.miscellaneous.model_reference:
            if not SharedModelManager.manager.miscellaneous.validate_model(
                model,
            ) and not SharedModelManager.manager.miscellaneous.download_model(model):
                logger.error(f"Failed to download model {model}")
                exit(1)
        else:
            logger.success("Downloaded all Miscellaneous models")

    if bridge_data.allow_post_processing:
        if SharedModelManager.manager.gfpgan is None:
            logger.error("Failed to load GFPGAN model manager")
            exit(1)
        if SharedModelManager.manager.esrgan is None:
            logger.error("Failed to load ESRGAN model manager")
            exit(1)
        if SharedModelManager.manager.codeformer is None:
            logger.error("Failed to load codeformer model manager")
            exit(1)

        SharedModelManager.manager.gfpgan.download_all_models()
        SharedModelManager.manager.gfpgan.download_all_models()
        for model in SharedModelManager.manager.gfpgan.model_reference:
            if not SharedModelManager.manager.gfpgan.validate_model(
                model,
            ) and not SharedModelManager.manager.gfpgan.download_model(model):
                logger.error(f"Failed to download model {model}")
                exit(1)
        else:
            logger.success("Downloaded all GFPGAN models")

        SharedModelManager.manager.esrgan.download_all_models()
        SharedModelManager.manager.esrgan.download_all_models()
        for model in SharedModelManager.manager.esrgan.model_reference:
            if not SharedModelManager.manager.esrgan.validate_model(
                model,
            ) and not SharedModelManager.manager.esrgan.download_model(model):
                logger.error(f"Failed to download model {model}")
                exit(1)
        else:
            logger.success("Downloaded all ESRGAN models")

        SharedModelManager.manager.codeformer.download_all_models()
        SharedModelManager.manager.codeformer.download_all_models()
        for model in SharedModelManager.manager.codeformer.model_reference:
            if not SharedModelManager.manager.codeformer.validate_model(
                model,
            ) and not SharedModelManager.manager.codeformer.download_model(model):
                logger.error(f"Failed to download model {model}")
                exit(1)

    if SharedModelManager.manager.compvis is None:
        logger.error("Failed to load compvis model manager")
        exit(1)

    from horde_worker_regen.model_download_core import ensure_models_present

    outcome = ensure_models_present(
        SharedModelManager.manager.compvis,
        list(bridge_data.image_models_to_load),
        on_model_finish=lambda name, _index, _total, ok: (
            None if ok else logger.error(f"Failed to download model {name}")
        ),
    )

    if outcome.failed:
        logger.error("Failed to download all models.")
        exit(1)
    else:
        logger.success("Downloaded all compvis (Stable Diffusion) models.")


def migrate_model_layout(*, apply: bool = False, load_config_from_env_vars: bool = False) -> int:
    """Migrate the on-disk model layout to the canonical horde_model_reference layout.

    A thin pass-through to horde_model_reference's ``migrate-model-layout``: it first loads the worker's
    environment (so ``AIWORKER_CACHE_HOME`` and any extra model directories are honored), then plans and,
    when *apply* is true, performs the symlink-safe relocation. Dry-run by default.
    """
    from horde_worker_regen.load_env_vars import load_env_vars_from_config

    if not load_config_from_env_vars:
        load_env_vars_from_config()

    from horde_model_reference.cli.migrate_layout import main as migrate_layout_main

    return migrate_layout_main(["--apply"] if apply else [])


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point: download all configured models, or migrate the on-disk layout."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="download_models",
        description="Download all models named in the worker config, or migrate the on-disk model layout.",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate the on-disk model layout instead of downloading (dry-run unless --apply is given).",
    )
    parser.add_argument("--apply", action="store_true", help="With --migrate, perform the moves (default: dry-run).")
    parser.add_argument(
        "--env",
        dest="load_config_from_env_vars",
        action="store_true",
        help="Load configuration from environment variables instead of the config file.",
    )
    parser.add_argument("--purge-unused-loras", action="store_true", help="Delete LoRAs not in the configured set.")
    parser.add_argument("--directml", type=int, default=None, help="DirectML device index (for Windows AMD GPUs).")
    args = parser.parse_args(argv)

    if args.migrate:
        raise SystemExit(
            migrate_model_layout(apply=args.apply, load_config_from_env_vars=args.load_config_from_env_vars),
        )

    download_all_models(
        load_config_from_env_vars=args.load_config_from_env_vars,
        purge_unused_loras=args.purge_unused_loras,
        directml=args.directml,
    )


if __name__ == "__main__":
    main()
