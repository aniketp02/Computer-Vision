@echo off

set /p repoPath="Enter repo to clone: "
git clone %repoPath%

rem taking input for the upstream file
set /p upstreamPath="Enter the Upstream path: "

rem add the upstream path
(git remote add upstream %upstreamPath%) || (echo Upstream already exists or enter valid path)

rem checking all the paths
echo Checking all the paths
git remote -v

git checkout main

rem pulling the changes
echo fetching files from upstream main
git fetch upstream main

rem merging the changes
echo merging files to main
(git merge upstream/main) || (
	echo mergin recursively
	git diff
	git merge -s recursive
)

pause
