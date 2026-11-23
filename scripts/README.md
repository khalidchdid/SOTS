# 📝 scripts

Scripts templates to ease app lifecycle, from creation to launch.

## Requirements

On windows, all you need is gitbash : https://git-scm.com/downloads
If, for some obscure reason, you don't want to use Gitbash but PowerShell, see this [section](#run-a-script-from-powershell).

## Run a script from gitbash

Use ``bash -i <path-to-script>``.

## Process from scratch

You can skip step 1 and 2 if already done on your system.
You can skip step 3 if you are working on an existing app.
You may skip step 4.

1. Install system wise configs
   **Windows/Linux**: add .condarc, .bashrc, .gitconfig and certificates in your home folder.
   **Windows**: add pip.ini config files in your appdata folder (Windows: %APPDATA%/pip/pip.ini).

   ```
   bash -i scripts/.setup-system.sh
   ```

   **TODO: Linux**: add pip.conf config file in your home folder ($HOME/.config/pip/pip.conf)
2. Install uv

   ```
   bash -i scripts/.install-uv.sh
   ```
3. Setup the project from templated script. Script will asks you for some metadata.

   ```
   bash -i scripts/.setup-project.sh
   ```
4. Install dependencies

   ```
   bash -i scripts/.install-app-deps.sh
   ```
5. Launch app

   ```
   bash -i scripts/.launch-app.sh
   ```

## Run a script from Powershell

If you absolutely insist on running the scripts from Windows PowerShell:
```powershell
.\scripts\.run-bash-script-on-ps.ps1 <path-to-script>
```
For example: `.\scripts\.run-bash-script-on-ps.ps1 .\scripts\.setup-project.sh`

## Run a script directly from scripts folder

Initially, this repository was conceived as a submodule. Each script therefore had to be executed in a parent folder of `scripts`. To allow scripts to be executed directly in the `scripts` folder, a dedicated script (`.run-script-from-here.sh`) was developed for this purpose. Here is how it works:
```bash
bash -i .run-script-from-here.sh <path-to-script>
```
For example: ``.run_script_from_here.sh .setup-system.sh``

## Use pytorch

If you want to add pytorch with CUDA GPU support to your project:

```bash
set -a
. .env
uv add torch==2.5.1+cu121
```

In above snippet, ```. .env``` import proxy env vars, and with ```set -a``` these vars are passed to uv which in turn can use proxies the right way. If you forget the ```set -a```, there is a high chance uv won't be able to add or sync. Also note that the torch version should be made explicit with the current Nexus way to this repository.

NB: ```set -a``` has been added to ```.bashrc.template```, so that it is always set to export all imported variables. Please, update your .bashrc accordingly, or run ```bash -i scripts/.setup-system.sh```script.


## ToDo

* [ ] App vs Lib templates
* [X] Handle pytorch gpu
* [X] License file
* [ ] Changelog file
* [ ] Tag on version on publish, disable publish/tag if version unchanged
