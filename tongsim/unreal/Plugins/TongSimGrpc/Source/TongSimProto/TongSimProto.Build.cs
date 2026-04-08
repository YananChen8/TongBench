using UnrealBuildTool;
using System;
using System.IO;
using System.Diagnostics;

public class TongSimProto : ModuleRules
{
	public TongSimProto(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
		bEnableExceptions = true;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"TongosGrpc",
				"TongSimMultiLevel",
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"Engine",

				// Temp RL Demo
				"TongSimVoxelGrid",
				"NavigationSystem",
				"AIModule",
				"TongSimCapture",
				"TongSimGameplay",
			}
		);

		GenerateProto();

		PrivateDefinitions.Add("SUPPRESS_PER_MODULE_INLINE_FILE"); // This module does not use core's standard operator new/delete overloads
	}

	private void GenerateProto()
	{
		// Resolve project and repository root:
		//   <repo_root>/
		//     ├─ protobuf/
		//     └─ unreal/ (PluginDirectory points inside this tree)
		string projectDir = Path.GetFullPath(Path.Combine(PluginDirectory, "..", ".."));
		string repoRoot = Path.GetFullPath(Path.Combine(projectDir, ".."));

		string repoProto = Path.Combine(repoRoot, "protobuf");
		string fallbackProto = Path.Combine(PluginDirectory, "Protobuf");
		string protoPath = Directory.Exists(repoProto) ? repoProto : fallbackProto;

		string genPath = Path.Combine(ModuleDirectory, "ThirdParty", "ProtoGen");
		Directory.CreateDirectory(genPath);
		if (!PrivateIncludePaths.Contains(genPath))
		{
			PrivateIncludePaths.Add(genPath);
		}

		string protocExe;
		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			protocExe = Path.Combine(PluginDirectory, "GrpcLibraries", "Win64", "protoc.exe");
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			protocExe = Path.Combine(PluginDirectory, "GrpcPrograms", "Linux", "protoc");
		}
		else
		{
			throw new BuildException("Unsupported platform for protoc generation.");
		}

		// Allow purely pre-generated sources if proto directory or protoc is missing.
		if (!Directory.Exists(protoPath))
		{
			Console.WriteLine($"[Protobuf] proto directory not found: {protoPath}. Using pre-generated sources if available.");
			return;
		}

		string[] allProtoFiles = Directory.GetFiles(protoPath, "*.proto", SearchOption.AllDirectories);
		if (allProtoFiles.Length == 0)
		{
			Console.WriteLine($"[Protobuf] No .proto files found under: {protoPath}. Using pre-generated sources if available.");
			return;
		}

		// Make sure UBT invalidates the makefile when any .proto changes.
		ExternalDependencies.AddRange(allProtoFiles);

		if (!File.Exists(protocExe))
		{
			Console.WriteLine($"[Protobuf] protoc not found: {protocExe}. Using pre-generated sources if available.");
			return;
		}

		string oldPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
		string newPath = Path.GetDirectoryName(protocExe) + Path.PathSeparator + oldPath;
		Environment.SetEnvironmentVariable("PATH", newPath);

		foreach (string protoFile in allProtoFiles)
		{
			string relativeProto = Path.GetRelativePath(protoPath, protoFile);
			string args = $"--proto_path=\"{protoPath}\" " +
					  $"--cpp_out=\"{genPath}\" " +
					  $"--grpc_cpp_out=\"{genPath}\" " +
					  $"\"{protoFile}\"";

			RunProtoc(protocExe, args);
			Console.WriteLine($"[Protobuf] Generated: {relativeProto}");
		}

		string descOut = Path.Combine(genPath, "all.proto.desc");
		string descArgs = $"--proto_path=\"{protoPath}\" " +
			              $"--descriptor_set_out=\"{descOut}\" --include_imports ";

		foreach (string protoFile in allProtoFiles)
		{
			descArgs += $"\"{protoFile}\" ";
		}

		RunProtoc(protocExe, descArgs);
		Console.WriteLine($"[Protobuf] Generated descriptor: {descOut}");

		WrapGeneratedProtoSources(genPath);

		/*
		string structOutPath = Path.Combine(ModuleDirectory, "Public", "AutoGenStructs");
		Directory.CreateDirectory(structOutPath);

		string includePrefix = "AutoGenStructs/";

		string codegenToolPath;
		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			codegenToolPath = Path.Combine(PluginDirectory, "Tools", "ProtoToUStructGenTool", "bin", "win", "ProtoToUStructGenTool.exe");
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			codegenToolPath = Path.Combine(PluginDirectory, "Tools", "ProtoToUStructGenTool", "bin", "linux", "ProtoToUStructGenTool");
		}
		else
		{
			throw new BuildException("Unsupported platform for ProtoToUstructGenTool");
		}

		if (!File.Exists(codegenToolPath))
		{
			throw new BuildException($"CodeGen tool not found: {codegenToolPath}");
		}

		string descPath = Path.Combine(genPath, "all.proto.desc");
		string codeGenArgs = $"\"{descPath}\" \"{structOutPath}\" \"{includePrefix}\"";

		RunCodeGenProcess(codegenToolPath, codeGenArgs);

		PublicIncludePaths.Add(structOutPath);
		*/
	}

	/// <summary>
	/// Wrap generated *.pb.cc files with warning push/pop pragmas.
	/// </summary>
	private void WrapGeneratedProtoSources(string genPath)
	{
		var ccFiles = Directory.GetFiles(genPath, "*.pb.cc", SearchOption.AllDirectories);
		foreach (var file in ccFiles)
		{
			string original = File.ReadAllText(file);

			var header =
@"#if defined(_MSC_VER)
	#pragma warning(push)
	#pragma warning(disable : 4800)      // uint64_t -> bool
#elif defined(__clang__) || defined(__GNUC__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored ""-Wint-conversion""
#endif

";
			var footer =
@"
#if defined(_MSC_VER)
	#pragma warning(pop)
#elif defined(__clang__) || defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif
";

			var tmp = file + ".tmp";
			File.WriteAllText(tmp, header + original + footer);
			File.Delete(file);
			File.Move(tmp, file);
			Console.WriteLine($"[Protobuf] Wrapped warnings in {file}");
		}
	}

	private void RunProtoc(string exe, string args)
	{
		ProcessStartInfo startInfo = new ProcessStartInfo(exe, args)
		{
			CreateNoWindow = true,
			UseShellExecute = false,
			RedirectStandardOutput = true,
			RedirectStandardError = true,
		};

		using (Process proc = Process.Start(startInfo))
		{
			string output = proc.StandardOutput.ReadToEnd();
			string error = proc.StandardError.ReadToEnd();
			proc.WaitForExit();

			if (!string.IsNullOrEmpty(output))
				Console.WriteLine("[protoc stdout] " + output);

			if (!string.IsNullOrEmpty(error))
				Console.WriteLine("[protoc stderr] " + error);

			if (proc.ExitCode != 0)
				throw new BuildException($"protoc failed: {error}");
		}
	}

	private void RunCodeGenProcess(string exe, string args)
	{
		var startInfo = new ProcessStartInfo(exe, args)
		{
			CreateNoWindow = true,
			UseShellExecute = false,
			RedirectStandardOutput = true,
			RedirectStandardError = true,
		};

		using (var proc = Process.Start(startInfo))
		{
			string stdout = proc.StandardOutput.ReadToEnd();
			string stderr = proc.StandardError.ReadToEnd();
			proc.WaitForExit();

			if (!string.IsNullOrWhiteSpace(stdout))
				Console.WriteLine("[CodeGen stdout] " + stdout);

			if (!string.IsNullOrWhiteSpace(stderr))
				Console.WriteLine("[CodeGen stderr] " + stderr);

			if (proc.ExitCode != 0)
				throw new BuildException($"CodeGen tool failed:\n{stderr}");
		}
	}
}
