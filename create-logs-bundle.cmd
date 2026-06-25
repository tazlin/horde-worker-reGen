@echo off
cd /d "%~dp0"
title AI Horde Worker

echo This will create a logs bundle for the Horde Worker. It will include the following:
echo - The worker's log files
echo - Your downloaded models and their metadata
echo - System information (OS, Python version, etc.)
echo All of the above will be anonymized, details such as system usernames in paths (like C:\Users\YourName) and API keys will be removed.
echo However, this is only a best-effort attempt, and you should review the bundle before sharing it if you have any concerns about sensitive information being included.
echo 
echo This may take a while...
call "%~dp0runtime.cmd" horde-log bundle %*
