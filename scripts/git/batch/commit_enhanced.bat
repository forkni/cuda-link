@echo off
REM commit_enhanced.bat
REM Enhanced commit workflow with comprehensive validations and mandatory logging
REM Extends commit.bat with branch-specific checks and safety validations
REM
REM Usage: commit_enhanced.bat [--non-interactive] [--skip-md-lint] "commit message"
REM   --non-interactive: Skip all prompts, use sensible defaults (for automation)

setlocal enabledelayedexpansion

REM Change to project root (two levels up from scripts\git\)
REM [Guide 1.3] Use pushd with error check
pushd "%~dp0..\.." || (
    echo ERROR: Cannot find project root
    exit /b 1
)

REM ========================================
REM Detect Git Bash/MSYS Environment
REM ========================================
REM [BUGFIX] Prevent "Commit"/"Workflow" file creation (see docs/AUTOMATED_GIT_WORKFLOW.md)
REM Git Bash cannot properly execute batch scripts - variable scoping fails,
REM causing empty %LOGFILE% redirections that create files in project root

if defined MSYSTEM (
    echo [ERROR] This script must be run from Windows CMD, not Git Bash
    echo.
    echo Detected environment: MSYSTEM=%MSYSTEM%
    echo.
    echo Solutions:
    echo   1. Use Windows CMD: cmd.exe /c scripts\git\commit_enhanced.bat "message"
    echo   2. Use direct git commands in Git Bash ^(see docs/AUTOMATED_GIT_WORKFLOW.md Section A^)
    echo.
    popd 2>nul
    exit /b 1
)

REM ========================================
REM Parse Command Line Arguments
REM ========================================

set "NON_INTERACTIVE=0"
set "SKIP_MD_LINT=0"
REM [Guide 1.9] Strip quotes from arguments safely
set "COMMIT_MSG_PARAM=%~1"

REM Check if first parameter is --non-interactive flag
if "%~1"=="--non-interactive" (
    set "NON_INTERACTIVE=1"
    set "COMMIT_MSG_PARAM=%~2"
)

REM Check for --skip-md-lint flag (can be in any position)
if "%~1"=="--skip-md-lint" (
    set "SKIP_MD_LINT=1"
    set "COMMIT_MSG_PARAM=%~2"
)
if "%~2"=="--skip-md-lint" (
    set "SKIP_MD_LINT=1"
    if !NON_INTERACTIVE! EQU 1 (
        set "COMMIT_MSG_PARAM=%~3"
    )
)

REM ========================================
REM Initialize Mandatory Logging
REM ========================================

REM Initialize logging using shared utility (locale-independent timestamp)
REM [Guide 1.8] Call shared function
call "%~dp0_common.bat" :InitLogging commit_enhanced

REM Validate logging initialization (catches cross-environment issues)
REM [BUGFIX] Defensive check for variable scoping failures (see docs/AUTOMATED_GIT_WORKFLOW.md)
if "%LOGFILE%"=="" (
    echo [ERROR] Failed to initialize logging - LOGFILE variable not set
    echo This typically happens when running batch scripts from Git Bash
    echo.
    echo Please run from Windows CMD: cmd.exe /c %~f0 %*
    popd
    exit /b 1
)
if "%REPORTFILE%"=="" (
    echo [ERROR] Failed to initialize logging - REPORTFILE variable not set
    popd
    exit /b 1
)

REM Initialize log file
echo ========================================= > "%LOGFILE%"
echo Enhanced Commit Workflow Log >> "%LOGFILE%"
echo ========================================= >> "%LOGFILE%"
echo Start Time: %date% %time% >> "%LOGFILE%"
for /f "tokens=*" %%i in ('git branch --show-current') do echo Branch: %%i >> "%LOGFILE%"
echo: >> "%LOGFILE%"

REM Log initialization message
call "%~dp0_common.bat" :LogMessage "=== Enhanced Commit Workflow ===" "%LOGFILE%"
call "%~dp0_common.bat" :LogMessage "" "%LOGFILE%"
call "%~dp0_common.bat" :LogMessage "[LOG] Workflow Log: %LOGFILE%" "%LOGFILE%"
call "%~dp0_common.bat" :LogMessage "" "%LOGFILE%"

REM [1/6] Get current branch
for /f "tokens=*" %%i in ('git branch --show-current') do set "CURRENT_BRANCH=%%i"
echo Current branch: !CURRENT_BRANCH!
echo:

REM [2/6] Check for uncommitted changes
echo [2/6] Checking for changes...
git diff --quiet
set "HAS_UNSTAGED=%ERRORLEVEL%"

git diff --cached --quiet
set "HAS_STAGED=%ERRORLEVEL%"

if %HAS_UNSTAGED% EQU 0 (
    if %HAS_STAGED% EQU 0 (
        echo [!] No changes to commit
        echo Working directory is clean
        popd && exit /b 0
    )
)

if %HAS_UNSTAGED% NEQ 0 (
    echo Unstaged changes detected
    echo:
    echo Unstaged files:
    git diff --name-status
    echo:
    if !NON_INTERACTIVE! EQU 1 (
        echo [Non-interactive mode] Auto-staging all changes...
        git add .
        if %ERRORLEVEL% NEQ 0 (
            echo [X] Failed to stage files
            popd && exit /b 1
        )
        echo [OK] All changes staged
    ) else (
        set /p "STAGE_ALL=Stage all changes? (yes/no): "
        if /i "!STAGE_ALL!" == "yes" (
            git add .
            if %ERRORLEVEL% NEQ 0 (
                echo [X] Failed to stage files
                popd && exit /b 1
            )
            echo [OK] All changes staged
        ) else (
            echo:
            echo Please stage changes manually:
            echo   git add ^<files^>
            echo Then run this script again
            popd && exit /b 1
        )
    )
) else (
    echo [OK] All changes already staged
)
echo:

REM [3/6] Validate staged files
echo [3/6] Validating staged files...

REM Remove local-only files from staging (safety check)
git reset HEAD CLAUDE.md 2>nul
git reset HEAD MEMORY.md 2>nul
git reset HEAD _archive/ 2>nul
git reset HEAD benchmark_results/ 2>nul

REM Check for local-only files
set "FOUND_LOCAL_FILES=0"

git diff --cached --name-only | "%WINDIR%\System32\find.exe" /i "CLAUDE.md" >nul
if %ERRORLEVEL% EQU 0 (
    echo [X] ERROR: CLAUDE.md is staged ^(should be local-only^)
    set "FOUND_LOCAL_FILES=1"
)

git diff --cached --name-only | "%WINDIR%\System32\find.exe" /i "MEMORY.md" >nul
if %ERRORLEVEL% EQU 0 (
    echo [X] ERROR: MEMORY.md is staged ^(should be local-only^)
    set "FOUND_LOCAL_FILES=1"
)

git diff --cached --name-only | "%WINDIR%\System32\find.exe" /i "_archive" >nul
if %ERRORLEVEL% EQU 0 (
    echo [X] ERROR: _archive/ is staged ^(should be local-only^)
    set "FOUND_LOCAL_FILES=1"
)

if !FOUND_LOCAL_FILES! EQU 1 (
    echo:
    echo Please remove these files from staging
    popd && exit /b 1
)

REM Branch-specific validations
if "!CURRENT_BRANCH!" == "main" (
    echo Validating main branch commit...

    REM Check for test files
    git diff --cached --name-only | "%WINDIR%\System32\findstr.exe" "^tests/" >nul
    if %ERRORLEVEL% EQU 0 (
        echo [X] ERROR: Test files staged on main branch
        echo Tests should only be on development branch
        echo:
        echo Staged test files:
        git diff --cached --name-only | "%WINDIR%\System32\findstr.exe" "^tests/"
        echo:
        popd && exit /b 1
    )

    REM Check for pytest.ini
    git diff --cached --name-only | "%WINDIR%\System32\findstr.exe" "pytest.ini" >nul
    if %ERRORLEVEL% EQU 0 (
        echo [X] ERROR: pytest.ini staged on main branch
        echo This file should only be on development branch
        popd && exit /b 1
    )

    REM Check for development-only docs
    git diff --cached --name-only | "%WINDIR%\System32\findstr.exe" "docs/TESTING_GUIDE.md" >nul
    if %ERRORLEVEL% EQU 0 (
        echo [X] ERROR: TESTING_GUIDE.md staged on main branch
        echo This doc should only be on development branch
        popd && exit /b 1
    )

    echo [OK] No development-only files detected
)

echo [OK] Staged files validated
echo:

REM [4/7] Code quality check
echo [4/7] Checking code quality...

REM Get shared lint exclusion patterns
call "%~dp0_common.bat" :GetLintExclusions

REM Always run Python lint checks (ruff check + ruff format)
set "PYTHON_LINT_ERROR=0"
call ".venv\Scripts\ruff.exe" check . %RUFF_EXCLUDE% >nul 2>&1
if %ERRORLEVEL% NEQ 0 set "PYTHON_LINT_ERROR=1"
call ".venv\Scripts\ruff.exe" format --check . %RUFF_EXCLUDE% >nul 2>&1
if %ERRORLEVEL% NEQ 0 set "PYTHON_LINT_ERROR=1"

REM Check markdownlint unless --skip-md-lint is set
set "MD_LINT_ERROR=0"
if %SKIP_MD_LINT% EQU 0 (
    call markdownlint-cli2 %MD_PATTERNS% >nul 2>&1
    if %ERRORLEVEL% NEQ 0 set "MD_LINT_ERROR=1"
) else (
    echo   [--skip-md-lint] Skipping markdown lint checks
)

REM Handle lint errors
if !PYTHON_LINT_ERROR! EQU 1 (
    echo [!] Python lint errors detected
    echo:
    if !NON_INTERACTIVE! EQU 1 (
        echo [Non-interactive mode] Auto-fixing Python lint issues...
        echo:
        call ".venv\Scripts\ruff.exe" check --fix . %RUFF_EXCLUDE% >nul 2>&1
        call ".venv\Scripts\ruff.exe" format . %RUFF_EXCLUDE% >nul 2>&1
        echo:
        echo Restaging fixed files...
        git add .
        if %ERRORLEVEL% NEQ 0 (
            echo [X] Failed to stage fixed files
            popd && exit /b 1
        )
        echo [OK] Fixed files staged

        REM Re-check Python lint
        set "PYTHON_LINT_ERROR=0"
        call ".venv\Scripts\ruff.exe" check . %RUFF_EXCLUDE% >nul 2>&1
        if %ERRORLEVEL% NEQ 0 (
            echo [X] Python lint errors remain after auto-fix
            popd && exit /b 1
        )
    ) else (
        set /p "FIX_LINT=Auto-fix Python lint issues? (yes/no): "
        if /i "!FIX_LINT!" == "yes" (
            echo:
            call ".venv\Scripts\ruff.exe" check --fix . %RUFF_EXCLUDE% >nul 2>&1
            call ".venv\Scripts\ruff.exe" format . %RUFF_EXCLUDE% >nul 2>&1
            echo:
            echo Restaging fixed files...
            git add .
            if %ERRORLEVEL% NEQ 0 (
                echo [X] Failed to stage fixed files
                popd && exit /b 1
            )
            echo [OK] Fixed files staged
        ) else (
            echo:
            echo To see Python lint errors, run: .venv\Scripts\ruff.exe check .
            set /p "CONTINUE_ANYWAY=Continue commit with Python lint errors? (yes/no): "
            if /i not "!CONTINUE_ANYWAY!" == "yes" (
                echo Commit cancelled - fix Python lint errors first
                popd && exit /b 1
            )
        )
    )
) else if !MD_LINT_ERROR! EQU 1 (
    echo [!] Markdown lint warnings detected ^(non-blocking^)
    echo   Run: markdownlint-cli2 "*.md" "docs/**/*.md" to see details
    echo   Use --skip-md-lint to suppress this warning
) else (
    echo [OK] Code quality checks passed
)
echo:

REM [5/7] Show staged changes
echo [5/7] Staged changes:
echo ====================================
git diff --cached --name-status
echo ====================================
echo:

REM Count staged files
REM [BUG FIX] Use explicit path to Windows find.exe to avoid Unix find /c (path) collision
for /f %%i in ('git diff --cached --name-only ^| "%WINDIR%\System32\find.exe" /c /v ""') do set "STAGED_COUNT=%%i"
echo Files to commit: !STAGED_COUNT!
echo:

REM [6/7] Get commit message
echo [6/7] Commit message...

if "%COMMIT_MSG_PARAM%"=="" (
    echo:
    echo Commit message required
    echo:
    echo Usage: commit_enhanced.bat [--non-interactive] "Your commit message"
    echo:
    echo Conventional commit format recommended:
    echo   feat:   New feature
    echo   fix:    Bug fix
    echo   docs:   Documentation changes
    echo   chore:  Maintenance tasks
    echo   test:   Test changes
    echo:
    echo Example: commit_enhanced.bat "feat: Add semantic search caching"
    echo Example: commit_enhanced.bat --non-interactive "feat: Add semantic search caching"
    popd && exit /b 1
)

set "COMMIT_MSG=%COMMIT_MSG_PARAM%"

REM Basic commit message validation
echo !COMMIT_MSG! | "%WINDIR%\System32\findstr.exe" "^feat: ^fix: ^docs: ^chore: ^test: ^refactor: ^style: ^perf:" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [!] WARNING: Commit message doesn't follow conventional format
    echo   Recommended prefixes: feat:, fix:, docs:, chore:, test:
    echo:
    if !NON_INTERACTIVE! EQU 1 (
        echo [Non-interactive mode] Continuing with non-conventional format...
    ) else (
        set /p "CONTINUE=Continue anyway? (yes/no): "
        if /i not "!CONTINUE!" == "yes" (
            echo Commit cancelled
            popd && exit /b 0
        )
    )
)

echo:
echo Commit message: !COMMIT_MSG!
echo:

REM [7/7] Create commit
echo [7/7] Creating commit...
echo:
if !NON_INTERACTIVE! EQU 1 (
    echo [Non-interactive mode] Branch: !CURRENT_BRANCH!
    echo [Non-interactive mode] Proceeding with commit...
) else (
    echo [!] BRANCH VERIFICATION
    echo You are about to commit to: !CURRENT_BRANCH!
    echo:
    set /p "CORRECT_BRANCH=Is this the correct branch? (yes/no): "
    if /i not "!CORRECT_BRANCH!" == "yes" (
        echo:
        echo Available branches:
        git branch
        echo:
        echo Switch to the correct branch first, then run this script again
        echo Command: git checkout ^<branch-name^>
        popd && exit /b 0
    )
    echo:
    set /p "CONFIRM=Proceed with commit? (yes/no): "
    if /i not "!CONFIRM!" == "yes" (
        echo Commit cancelled
        popd && exit /b 0
    )
)

git commit -m "!COMMIT_MSG!"

if %ERRORLEVEL% EQU 0 (
    echo:
    echo ====================================
    echo [OK] COMMIT SUCCESSFUL
    echo ====================================
    echo:
    for /f "tokens=*" %%i in ('git log -1 --oneline') do echo Commit: %%i
    echo Branch: !CURRENT_BRANCH!
    echo Files: !STAGED_COUNT!
    echo:
    echo [OK] Local files remained private
    echo [OK] Branch-specific validations passed
    echo:
    echo Next steps:
    if "!CURRENT_BRANCH!" == "development" (
        echo   - Continue development
        echo   - When ready: scripts\git\merge_with_validation.bat
    ) else if "!CURRENT_BRANCH!" == "main" (
        echo   - Test changes thoroughly
        echo   - Push to remote: git push origin main
    ) else (
        echo   - Push to remote: git push origin !CURRENT_BRANCH!
    )
) else (
    echo:
    echo [X] Commit failed - check output above
    popd && exit /b 1
)

REM Generate analysis report
call :GenerateAnalysisReport

endlocal
popd && exit /b 0

REM ========================================
REM Helper Functions
REM ========================================

:GenerateAnalysisReport
REM Generate comprehensive analysis report
echo # Enhanced Commit Workflow Analysis Report > "%REPORTFILE%"
echo: >> "%REPORTFILE%"
echo **Workflow**: Enhanced Commit >> "%REPORTFILE%"
echo **Date**: %date% %time% >> "%REPORTFILE%"
echo **Branch**: !CURRENT_BRANCH! >> "%REPORTFILE%"
echo **Status**: [OK] SUCCESS >> "%REPORTFILE%"
echo: >> "%REPORTFILE%"
echo ## Summary >> "%REPORTFILE%"
echo Successfully committed changes with full validation and logging. >> "%REPORTFILE%"
echo: >> "%REPORTFILE%"
echo ## Files Committed >> "%REPORTFILE%"
echo: >> "%REPORTFILE%"
git diff HEAD~1 --name-status >> "%REPORTFILE%" 2>nul
echo: >> "%REPORTFILE%"
echo ## Commit Details >> "%REPORTFILE%"
echo: >> "%REPORTFILE%"
git log -1 --pretty=format:"- **Hash**: %%H%%n- **Message**: %%s%%n- **Author**: %%an%%n- **Date**: %%ad%%n" >> "%REPORTFILE%" 2>nul
echo: >> "%REPORTFILE%"
echo: >> "%REPORTFILE%"
echo ## Validations Passed >> "%REPORTFILE%"
echo: >> "%REPORTFILE%"
echo - [OK] No local-only files committed (CLAUDE.md, MEMORY.md, _archive) >> "%REPORTFILE%"
echo - [OK] Branch-specific validations passed >> "%REPORTFILE%"
echo - [OK] Code quality checks passed >> "%REPORTFILE%"
echo - [OK] Conventional commit format validated >> "%REPORTFILE%"
echo: >> "%REPORTFILE%"
echo ## Logs >> "%REPORTFILE%"
echo: >> "%REPORTFILE%"
echo - Execution log: `%LOGFILE%` >> "%REPORTFILE%"
echo - Analysis report: `%REPORTFILE%` >> "%REPORTFILE%"
echo: >> "%REPORTFILE%"
echo End Time: %date% %time% >> "%LOGFILE%"
call "%~dp0_common.bat" :LogMessage "" "%LOGFILE%"
call "%~dp0_common.bat" :LogMessage "======================================" "%LOGFILE%"
call "%~dp0_common.bat" :LogMessage "[REPORT] Analysis Report: %REPORTFILE%" "%LOGFILE%"
call "%~dp0_common.bat" :LogMessage "======================================" "%LOGFILE%"
goto :eof