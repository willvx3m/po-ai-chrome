# Automate Hourly Screenshots on Windows

To take a screenshot every hour in Windows, you can use a combination of the built-in Snipping Tool (or Snip & Sketch) and Windows Task Scheduler to automate the process. Below is a step-by-step guide:

## Solution: Automate Screenshots Using PowerShell and Task Scheduler

### Step 1: Create a PowerShell Script

1. **Open Notepad or any text editor.**

2. **Copy and paste the following PowerShell script to capture a screenshot and save it with a timestamp:**

   ```powershell
   $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
   $filePath = "C:\Screenshots\screenshot_$timestamp.png"
   Add-Type -AssemblyName System.Windows.Forms
   Add-Type -AssemblyName System.Drawing
   $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
   $bitmap = New-Object System.Drawing.Bitmap $screen.Width, $screen.Height
   $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
   $graphics.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
   $bitmap.Save($filePath, [System.Drawing.Imaging.ImageFormat]::Png)
   $graphics.Dispose()
   $bitmap.Dispose()
   ```

3. **Save the script:**
   - Save it as `screenshot.ps1` in a folder, e.g., `C:\Scripts\`.
   - Ensure the folder `C:\Screenshots\` exists (create it manually if needed, as this is where screenshots will be saved).

### Step 2: Set Up Task Scheduler

1. **Open Task Scheduler:**
   - Press `Win + S`, type Task Scheduler, and open it.

2. **Create a New Task:**
   - Click `Action > Create Task` (not "Create Basic Task" for more control).

3. **In the General tab:**
   - Name the task, e.g., `Hourly Screenshot`.
   - Check `Run whether user is logged on or not` if you want it to run in the background.
   - Check `Run with highest privileges`.

4. **Set the Trigger:**
   - Go to the `Triggers` tab, click `New`.
   - Set `Begin the task` to `On a schedule`.
   - Choose `Daily`, and set `Recur every 1 days`.
   - Under `Advanced settings`, check `Repeat task every` and select `1 hour` from the dropdown.
   - Set `for a duration of` to `Indefinitely`.
   - Click `OK`.

5. **Set the Action:**
   - Go to the `Actions` tab, click `New`.
   - Set `Action` to `Start a program`.
   - In `Program/script`, enter: `powershell.exe`
   - In `Add arguments (optional)`, enter: `-ExecutionPolicy Bypass -File "C:\Scripts\screenshot.ps1"`
   - Click `OK`.

6. **Set Conditions and Settings (optional):**
   - In the `Conditions` tab, uncheck `Start the task only if the computer is on AC power` if you want it to run on a laptop battery.
   - In the `Settings` tab, ensure `Allow task to be run on demand` is checked and adjust other settings as needed.

7. **Save the Task:**
   - Click `OK` to save the task.
   - If prompted, enter your user credentials (username and password) to allow the task to run when not logged in.

### Step 3: Test the Script

1. **In Task Scheduler, find your task in the Task Scheduler Library, right-click it, and select Run.**

2. **Check the `C:\Screenshots\` folder to verify that a screenshot was saved with a timestamp.**

### Notes

- **PowerShell Execution Policy:** If you encounter issues running the script, open PowerShell as Administrator and run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Bypass` to allow script execution.

- **Folder Permissions:** Ensure the `C:\Screenshots\` folder has write permissions for the user account running the task.

- **Alternative Tools:** If you prefer a GUI-based tool, third-party software like AutoHotkey or ShareX can also automate screenshots but may require more setup.

This setup will take a screenshot every hour and save it to `C:\Screenshots\` with a timestamp. Let me know if you need help troubleshooting or customizing this further!

