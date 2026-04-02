@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found: .venv
  echo Run first:
  echo   python -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install -r requirements.txt
  exit /b 1
)

if not exist "ntcnpdf.pdf" (
  echo [ERROR] Missing file: ntcnpdf.pdf
  exit /b 1
)

if not exist "Example.xlsx" (
  echo [ERROR] Missing file: Example.xlsx
  exit /b 1
)

for /f "usebackq delims=" %%i in (".env") do (
  rem placeholder loop to force no-op if .env exists
)

if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" if not "%%A:~0,1"=="#" set "%%A=%%B"
  )
)

".venv\Scripts\python.exe" oil_batch_mvp.py --pdf .\ntcnpdf.pdf --batch-db .\Example.xlsx --out .\result.xlsx

if errorlevel 1 (
  echo [ERROR] Script failed.
  exit /b 1
)

echo [OK] Done. Output file: result.xlsx
endlocal
