@echo off
echo [CLEAN] Removing generated files...

REM 删除 ProtoGen 目录下所有内容
IF EXIST "%~dp0Source\TongSimProto\ThirdParty\ProtoGen" (
    rmdir /s /q "%~dp0Source\TongSimProto\ThirdParty\ProtoGen"
    echo [CLEAN] Deleted: ThirdParty\ProtoGen
)

REM 删除 AutoGenStructs（整个文件夹）
IF EXIST "%~dp0Source\TongSimProto\Public\AutoGenStructs" (
    rmdir /s /q "%~dp0Source\TongSimProto\Public\AutoGenStructs"
    echo [CLEAN] Deleted: AutoGenStructs
)

echo [CLEAN] Done.
pause
