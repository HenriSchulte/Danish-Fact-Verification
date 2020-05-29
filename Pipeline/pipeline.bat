@echo off
@setlocal EnableDelayedExpansion

set claim=""
set label=""
if "%1" == "-c" (
	echo Write a claim:
	set /p claim=
	echo Pick a label [Refuted, Supported, NotEnoughInfo]:
	set /p label=
	java -jar DocumentRetrieval.jar "!claim!" !label!
)
if "%1" == "-f" (
	java -jar DocumentRetrieval.jar "%2" "%3" "%4"
)
python classifier.py
echo Done.