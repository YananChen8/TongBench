using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.RegularExpressions;
using UnrealBuildTool;

public class TongosGrpc : ModuleRules
{
	private UnrealTargetPlatform Platform;
	private UnrealTargetConfiguration Configuration;

	// name of root folders in the project folder
	private static readonly string GRPC_STRIPPED_FOLDER = "GrpcIncludes";
	private static readonly string GRPC_LIBS_FOLDER = "GrpcLibraries";

	private string INCLUDE_ROOT;
	private string LIB_ROOT;

	public class ModuleDepPaths
	{
		public readonly string[] HeaderPaths;
		public readonly string[] LibraryPaths;

		public ModuleDepPaths(string[] headerPaths, string[] libraryPaths)
		{
			HeaderPaths = headerPaths;
			LibraryPaths = libraryPaths;
		}

		public override string ToString()
		{
			return "Headers:\n" + string.Join("\n", HeaderPaths) + "\nLibs:\n" + string.Join("\n", LibraryPaths);
		}
	}

	[Conditional("DEBUG")]
	[Conditional("TRACE")]
	private void clog(params object[] objects)
	{
		Console.WriteLine(string.Join(", ", objects));
	}

	private IEnumerable<string> FindFilesInDirectory(string dir, string suffix = "")
	{
		List<string> matches = new List<string>();
		if (Directory.Exists(dir))
		{
			string[] files = Directory.GetFiles(dir);
			Regex regex = new Regex(".+\\." + suffix + "$");

			foreach (string file in files)
			{
				if (regex.Match(file).Success)
					matches.Add(file);
			}
		}

		return matches;
	}

    private string GetConfigurationString()
	{
		return (Configuration == UnrealTargetConfiguration.Shipping) ? "Release" : "Debug";
	}

	public ModuleDepPaths GatherDeps()
	{
		string RootPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "../../"));

		INCLUDE_ROOT = Path.Combine(RootPath, GRPC_STRIPPED_FOLDER);
		LIB_ROOT = Path.Combine(RootPath, GRPC_LIBS_FOLDER);


		List<string> headers = new List<string>();
		List<string> libs = new List<string>();

		string PlatformLibRoot = "";


		if (Platform == UnrealTargetPlatform.Win64)
		{
			PlatformLibRoot = Path.Combine(LIB_ROOT, Platform.ToString());
			libs.AddRange(FindFilesInDirectory(PlatformLibRoot, "lib"));
            foreach (var dllLib in FindFilesInDirectory(PlatformLibRoot, "dll")) {
                string[] segs =  dllLib.Split("\\");
				string dllName = segs[segs.Length - 1];
                RuntimeDependencies.Add("$(BinaryOutputDir)/" +dllName, dllLib);
                Console.WriteLine("tongosgrpc dll lib:" + dllLib);
            }
        }
		else if (Platform == UnrealTargetPlatform.Linux)
		{
			PlatformLibRoot = Path.Combine(LIB_ROOT, "Linux");
			libs.AddRange(FindFilesInDirectory(PlatformLibRoot, "so"));
		}
		PrivateRuntimeLibraryPaths.Add(PlatformLibRoot);

		Console.WriteLine("PlatformLibRoot: " + PlatformLibRoot);
		clog("PlatformLibRoot: " + PlatformLibRoot);

		headers.Add(Path.Combine(INCLUDE_ROOT, "include"));
		// headers.Add(Path.Combine(INCLUDE_ROOT, "third_party", "protobuf", "src"));

		return new ModuleDepPaths(headers.ToArray(), libs.ToArray());
	}

	public TongosGrpc(ReadOnlyTargetRules Target) : base(Target)
	{
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		CppCompileWarningSettings.UndefinedIdentifierWarningLevel = WarningLevel.Off;
		bEnableExceptions = true;
		PublicDefinitions.Add("GOOGLE_PROTOBUF_INTERNAL_DONATE_STEAL_INLINE=0");
		PublicDefinitions.Add("GOOGLE_PROTOBUF_NO_RTTI");
		PublicDefinitions.Add("GPR_FORBID_UNREACHABLE_CODE=0");
		PublicDefinitions.Add("GRPC_ALLOW_EXCEPTIONS=0");
		PublicDefinitions.Add("PROTOBUF_ENABLE_DEBUG_LOGGING_MAY_LEAK_PII=0");

        PublicDefinitions.Add("PROTOBUF_USE_DLLS");
		if(Target.Platform == UnrealTargetPlatform.Win64){
			PublicDefinitions.Add("gRPCXX_DLL_IMPORTS");
			PublicDefinitions.Add("ABSL_CONSUME_DLL");
		} else {
			//Linux
			PublicDefinitions.Add("GRPCXX_DLL=");
		}

		PublicDefinitions.Add("absl=absl_tong");
		PublicDefinitions.Add("grpc_error_to_absl_status=grpc_error_to_absl_tong_status");
		PublicDefinitions.Add("absl_random_internal_seed_material=absl_tong_random_internal_seed_material");
		PublicDefinitions.Add("AbslContainerInternalSampleEverything=AbslTongContainerInternalSampleEverything");
		PublicDefinitions.Add("AbslInternalGetFileMappingHint=AbslTongInternalGetFileMappingHint");
		PublicDefinitions.Add("AbslInternalMutexYield=AbslTongInternalMutexYield");
		PublicDefinitions.Add("AbslInternalPerThreadSemPost=AbslTongInternalPerThreadSemPost");
		PublicDefinitions.Add("AbslInternalPerThreadSemWait=AbslTongInternalPerThreadSemWait");
		PublicDefinitions.Add("AbslInternalReportFatalUsageError=AbslTongInternalReportFatalUsageError");
		PublicDefinitions.Add("AbslInternalSleepFor=AbslTongInternalSleepFor");
		PublicDefinitions.Add("AbslInternalSpinLockDelay=AbslTongInternalSpinLockDelay");
		PublicDefinitions.Add("AbslInternalSpinLockWake=AbslTongInternalSpinLockWake");

		CppCompileWarningSettings.UndefinedIdentifierWarningLevel = WarningLevel.Off;
		//TODO: We do this because in file generated_message_table_driven.h that located in protobuf sources
		//TODO: line 174: static_assert(std::is_pod<AuxillaryParseTableField>::value, "");
		//TODO: causes ��4647 level 3 warning __is_pod behavior change
		//TODO: UE4 threading some warnings as errors, and we have no chance to suppress this stuff
		//TODO: So, we don't want to change any third-party code, this why we add this definition
		PublicDefinitions.Add("__NVCC__");

		Platform = Target.Platform;
		Configuration = Target.Configuration;

		ModuleDepPaths moduleDepPaths = GatherDeps();
		Console.WriteLine(moduleDepPaths.ToString());

		PublicIncludePaths.AddRange(moduleDepPaths.HeaderPaths);

		PublicAdditionalLibraries.InsertRange(0, moduleDepPaths.LibraryPaths);
		if (Platform == UnrealTargetPlatform.Linux)
		{
			// Fix BuildPipeline Path Cannot Start With "/"
			PublicAdditionalLibraries.Add(Path.Combine(LIB_ROOT, "lib64/libc.a"));
            foreach (var libPath in moduleDepPaths.LibraryPaths)
            {
                RuntimeDependencies.Add(libPath);
            }
        }

		foreach (var lib in PublicAdditionalLibraries)
		{
			Console.WriteLine("tongosgrpc path:" + lib);
		}

        PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"Json",
			"TongSimMemoryFixer"
		});

		AddEngineThirdPartyPrivateStaticDependencies(Target, "OpenSSL");
		AddEngineThirdPartyPrivateStaticDependencies(Target, "zlib");

		PrivateDependencyModuleNames.AddRange(new string[] { "CoreUObject", "Engine", "DeveloperSettings" });


		if (Platform == UnrealTargetPlatform.Win64)
		{
			string BaseDirectory = Path.GetFullPath(Path.Combine(ModuleDirectory, "..", ".."));
			string LibrariesDirectory = Path.Combine(BaseDirectory, "DynamicLibraries/Win64");
			PublicDelayLoadDLLs.Add("zlib.dll");
			RuntimeDependencies.Add(Path.Combine(LibrariesDirectory, "zlib.dll"));
		}

		PrivateDefinitions.Add("SUPPRESS_PER_MODULE_INLINE_FILE"); // This module does not use core's standard operator new/delete overloads
	}
}
