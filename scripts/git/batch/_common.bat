@echo off
REM _common.bat - Shared utility functions for git automation scripts
REM Purpose: Centralized functions for timestamp generation, logging, and lint configuration
REM Usage: call "%~dp0_common.bat" :FunctionName [args...]
REM
REM Available Functions:
REM   :GetTimestamp           - Sets TIMESTAMP variable (yyyyMMdd_HHmmss format)
REM   :InitLogging            - Sets LOGFILE, REPORTFILE variables and creates log directory
REM   :GetLintExclusions      - Sets lint exclusion variables for all tools
REM   :LogMessage             - Logs message to console and file
REM
REM Compliance: BATCH_STYLE_GUIDE.md Section 1 (Core Patterns)

REM Route to requested function
if "%~1"=="" goto :eof
call %~1 %~2 %~3 %~4 %~5 %~6 %~7 %~8 %~9
goto :eof

REM ========================================
REM Function: GetTimestamp
REM ========================================
REM Purpose: Generate locale-independent timestamp
REM Returns: Sets TIMESTAMP variable to yyyyMMdd_HHmmss format
REM Example: call "%~dp0_common.bat" :GetTimestamp
REM          echo %TIMESTAMP% -> 20251203_143025
REM
REM Note: Uses PowerShell for consistent formatting across all locales
REM       Replaces locale-dependent date /t and time /t parsing
:GetTimestamp
for /f "usebackq" %%i in (`powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"`) do set "TIMESTAMP=%%i"
goto :eof

REM ========================================
REM Function: InitLogging
REM ========================================
REM Purpose: Initialize logging infrastructure for script
REM Args: %1 = script_name (e.g., "commit_enhanced", "check_lint")
REM Returns: Sets LOGFILE and REPORTFILE variables, creates logs/ directory
REM Example: call "%~dp0_common.bat" :InitLogging commit_enhanced
REM          echo %LOGFILE% -> logs\commit_enhanced_20251203_143025.log
:InitLogging
if not exist "logs\" mkdir "logs"
call :GetTimestamp
set "LOGFILE=logs\%~1_%TIMESTAMP%.log"
set "REPORTFILE=logs\%~1_analysis_%TIMESTAMP%.md"
goto :eof

REM ========================================
REM Function: GetLintExclusions
REM ========================================
REM Purpose: Set consistent exclusion patterns for all lint tools
REM Returns: Sets RUFF_EXCLUDE, MD_PATTERNS variables
REM Example: call "%~dp0_common.bat" :GetLintExclusions
REM          echo %RUFF_EXCLUDE% -> --extend-exclude tests/test_data --extend-exclude _archive
REM
REM Exclusion Rationale:
REM   - tests/test_data: Contains intentional lint errors for testing
REM   - _archive: Historical code not subject to current standards
:GetLintExclusions
set "RUFF_EXCLUDE=--extend-exclude tests/test_data --extend-exclude _archive"
set "MD_PATTERNS=*.md .claude/**/*.md .github/**/*.md .githooks/**/*.md .vscode/**/*.md docs/**/*.md tests/**/*.md"
goto :eof

REM ========================================
REM Function: LogMessage
REM ========================================
REM Purpose: Log message to both console and file
REM Args: %1 = message text, %2 = LOGFILE path
REM Example: call "%~dp0_common.bat" :LogMessage "Processing files..." "%LOGFILE%"
:LogMessage
set "MSG=%~1"
echo %MSG%
echo %MSG% >> "%~2"
goto :eof
