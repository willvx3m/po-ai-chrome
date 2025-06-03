# Use a Batch File with a Loop

You can create a batch file that runs the PowerShell script in a loop with a delay, effectively scheduling it every hour.

## Steps:

### Create the PowerShell Script:

Ensure your script (e.g., `C:\Scripts\screenshot.ps1`) is ready, as provided in the original response.

### Create a Batch File:

Open Notepad and create a file with the following content:

```batch
@echo off
:loop
powershell.exe -ExecutionPolicy Bypass -File "C:\Scripts\screenshot.ps1"
timeout /t 3600 /nobreak
goto loop
```

Save it as `run_screenshot.bat` in a folder (e.g., `C:\Scripts\`).

The `timeout /t 3600` command pauses for 3,600 seconds (1 hour).

### Run the Batch File:

Double-click the batch file to start it, or run it from the Command Prompt.

To run it in the background, create a shortcut to the batch file, right-click the shortcut, go to Properties, and set Run to Minimized.

