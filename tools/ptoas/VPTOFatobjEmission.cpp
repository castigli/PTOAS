// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOFatobjEmission.h"

#include "PTO/Transforms/VPTOLLVMEmitter.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cctype>
#include <optional>
#include <string>

namespace {

using llvm::StringRef;

static bool runCommandWithStderr(llvm::StringRef program,
                                 llvm::ArrayRef<std::string> ownedArgs,
                                 llvm::StringRef stderrPath,
                                 llvm::raw_ostream &diagOS,
                                 llvm::StringRef what,
                                 std::optional<llvm::StringRef> stdinPath =
                                     std::nullopt);

class TempFileRegistry {
public:
  ~TempFileRegistry() { cleanup(); }

  void cleanup() {
    for (const std::string &path : paths)
      llvm::sys::fs::remove(path);
    paths.clear();
  }

  bool create(StringRef prefix, StringRef suffix, std::string &path,
              llvm::raw_ostream &diagOS) {
    llvm::SmallString<128> tempPath;
    int fd = -1;
    std::error_code ec = llvm::sys::fs::createTemporaryFile(prefix, suffix, fd,
                                                            tempPath);
    if (ec) {
      diagOS << "Error: failed to create temporary file for " << prefix
             << suffix << ": " << ec.message() << "\n";
      return false;
    }
    llvm::sys::Process::SafelyCloseFileDescriptor(fd);
    path = tempPath.str().str();
    paths.push_back(path);
    return true;
  }

private:
  llvm::SmallVector<std::string, 8> paths;
};

static bool writeTextFile(StringRef path, StringRef content,
                          llvm::raw_ostream &diagOS) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    diagOS << "Error: failed to open " << path << " for write: "
           << ec.message() << "\n";
    return false;
  }
  os << content;
  os.flush();
  return true;
}

static bool writeLLVMModuleFile(llvm::Module &module, StringRef path,
                                llvm::raw_ostream &diagOS) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    diagOS << "Error: failed to open " << path << " for write: "
           << ec.message() << "\n";
    return false;
  }
  module.print(os, nullptr);
  os.flush();
  return true;
}

static std::string sanitizeModuleId(llvm::StringRef raw) {
  std::string out;
  out.reserve(raw.size());
  for (char c : raw) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
      out.push_back(c);
    else
      out.push_back('_');
  }
  if (out.empty())
    out = "ptoas_fatobj";
  return out;
}

static std::optional<std::string> getAscendHomePath() {
  const char *env = std::getenv("ASCEND_HOME_PATH");
  if (!env || !*env)
    return std::nullopt;
  return std::string(env);
}

static std::string joinPath(llvm::StringRef lhs, llvm::StringRef rhs) {
  llvm::SmallString<256> joined(lhs);
  llvm::sys::path::append(joined, rhs);
  return std::string(joined.str());
}

static std::optional<std::string> locateProgram(llvm::StringRef envPath,
                                                llvm::StringRef fallbackName) {
  if (!envPath.empty() && llvm::sys::fs::exists(envPath))
    return envPath.str();
  if (auto found = llvm::sys::findProgramByName(fallbackName))
    return *found;
  return std::nullopt;
}

class VPTOFatobjToolchain;

static bool compileDeviceLLVMToObject(llvm::StringRef llPath,
                                      llvm::StringRef outObjPath,
                                      llvm::StringRef targetCPU,
                                      llvm::StringRef bishengPath,
                                      llvm::StringRef stderrPath,
                                      llvm::raw_ostream &diagOS);
static bool compileHostStubToFatobj(llvm::StringRef stubPath,
                                    llvm::StringRef outObjPath,
                                    llvm::StringRef moduleId,
                                    llvm::StringRef targetCPU,
                                    const VPTOFatobjToolchain &toolchain,
                                    llvm::StringRef deviceObjPath,
                                    llvm::StringRef stderrPath,
                                    llvm::raw_ostream &diagOS);
static bool mergeDeviceObjects(llvm::ArrayRef<std::string> deviceObjPaths,
                               llvm::StringRef outObjPath,
                               llvm::StringRef ldLldPath,
                               llvm::StringRef stderrPath,
                               llvm::raw_ostream &diagOS);

class VPTOFatobjToolchain {
public:
  static std::optional<VPTOFatobjToolchain>
  create(llvm::raw_ostream &diagOS) {
    std::optional<std::string> ascendHome = getAscendHomePath();
    if (!ascendHome) {
      diagOS << "Error: ASCEND_HOME_PATH is required for VPTO fatobj emission.\n";
      return std::nullopt;
    }

    VPTOFatobjToolchain toolchain(*ascendHome);
    if (!toolchain.validate(diagOS))
      return std::nullopt;
    return toolchain;
  }

  const std::string &ascendHome() const { return ascendHomePath; }
  const std::string &bisheng() const { return bishengPath; }
  const std::string &bishengCc1() const { return bishengCc1Path; }
  const std::string &cceLd() const { return cceLdPath; }
  const std::string &ldLld() const { return ldLldPath; }
  const std::string &resourceDir() const { return resourceDirPath; }
  const std::string &resourceIncludeDir() const {
    return resourceIncludeDirPath;
  }
  const std::string &cceStubDir() const { return cceStubDirPath; }
  const std::string &bishengCompilerBinDir() const {
    return bishengCompilerBinDirPath;
  }

private:
  explicit VPTOFatobjToolchain(llvm::StringRef ascendHome)
      : ascendHomePath(ascendHome.str()),
        bishengPath(joinPath(ascendHomePath, "bin/bisheng")),
        bishengCc1Path(
            joinPath(ascendHomePath, "tools/bisheng_compiler/bin/bisheng")),
        cceLdPath(joinPath(ascendHomePath, "bin/cce-ld")),
        ldLldPath(
            locateProgram(joinPath(ascendHomePath, "bin/ld.lld"), "ld.lld")
                .value_or(std::string())),
        resourceDirPath(joinPath(
            ascendHomePath, "tools/bisheng_compiler/lib/clang/15.0.5")),
        resourceIncludeDirPath(joinPath(resourceDirPath, "include")),
        cceStubDirPath(joinPath(resourceIncludeDirPath, "cce_stub")),
        bishengCompilerBinDirPath(
            joinPath(ascendHomePath, "tools/bisheng_compiler/bin")) {}

  bool validate(llvm::raw_ostream &diagOS) const {
    if (!llvm::sys::fs::exists(bishengPath)) {
      diagOS << "Error: unable to locate bisheng: " << bishengPath << "\n";
      return false;
    }
    if (!llvm::sys::fs::exists(bishengCc1Path)) {
      diagOS << "Error: unable to locate bisheng cc1 frontend: "
             << bishengCc1Path << "\n";
      return false;
    }
    if (!llvm::sys::fs::exists(cceLdPath)) {
      diagOS << "Error: unable to locate cce-ld: " << cceLdPath << "\n";
      return false;
    }
    if (ldLldPath.empty() || !llvm::sys::fs::exists(ldLldPath)) {
      diagOS << "Error: unable to locate ld.lld.\n";
      return false;
    }
    return true;
  }

  std::string ascendHomePath;
  std::string bishengPath;
  std::string bishengCc1Path;
  std::string cceLdPath;
  std::string ldLldPath;
  std::string resourceDirPath;
  std::string resourceIncludeDirPath;
  std::string cceStubDirPath;
  std::string bishengCompilerBinDirPath;
};

class VPTOFatobjArtifacts {
public:
  explicit VPTOFatobjArtifacts(TempFileRegistry &tempFiles)
      : tempFiles(tempFiles) {}

  bool emitStubSource(StringRef stubSource, llvm::raw_ostream &diagOS) {
    if (!tempFiles.create("ptoas-host-stub", ".cpp", stubPath, diagOS))
      return false;
    if (!writeTextFile(stubPath, stubSource, diagOS))
      return false;
    return true;
  }

  bool initCommandLogs(llvm::raw_ostream &diagOS) {
    if (!tempFiles.create("ptoas-stderr", ".log", stderrPath, diagOS))
      return false;
    return true;
  }

  bool emitCubeObject(llvm::Module *module,
                      const VPTOFatobjToolchain &toolchain,
                      llvm::StringRef targetCPU,
                      llvm::raw_ostream &diagOS) {
    if (!module)
      return true;
    if (!tempFiles.create("ptoas-device", ".ll", cubeLLPath, diagOS))
      return false;
    if (!writeLLVMModuleFile(*module, cubeLLPath, diagOS))
      return false;
    if (!tempFiles.create("ptoas-device", ".o", cubeObjPath, diagOS))
      return false;
    return compileDeviceLLVMToObject(cubeLLPath, cubeObjPath,
                                     targetCPU, toolchain.bisheng(),
                                     stderrPath, diagOS);
  }

  bool emitVectorObject(llvm::Module *module,
                        const VPTOFatobjToolchain &toolchain,
                        llvm::StringRef targetCPU,
                        llvm::raw_ostream &diagOS) {
    if (!module)
      return true;
    if (!tempFiles.create("ptoas-device", ".ll", vectorLLPath, diagOS))
      return false;
    if (!writeLLVMModuleFile(*module, vectorLLPath, diagOS))
      return false;
    if (!tempFiles.create("ptoas-device", ".o", vectorObjPath, diagOS))
      return false;
    return compileDeviceLLVMToObject(vectorLLPath, vectorObjPath,
                                     targetCPU, toolchain.bisheng(),
                                     stderrPath, diagOS);
  }

  bool mergeDeviceObjects(const VPTOFatobjToolchain &toolchain,
                          llvm::raw_ostream &diagOS) {
    llvm::SmallVector<std::string, 2> deviceObjPaths;
    if (!cubeObjPath.empty())
      deviceObjPaths.push_back(cubeObjPath);
    if (!vectorObjPath.empty())
      deviceObjPaths.push_back(vectorObjPath);
    if (deviceObjPaths.empty()) {
      diagOS << "Error: VPTO fatobj emission requires at least one device module.\n";
      return false;
    }
    if (!tempFiles.create("ptoas-device-merged", ".o", mergedDeviceObjPath,
                          diagOS))
      return false;
    return ::mergeDeviceObjects(deviceObjPaths, mergedDeviceObjPath,
                                toolchain.ldLld(), stderrPath, diagOS);
  }

  bool compileHostStub(const VPTOFatobjToolchain &toolchain,
                       llvm::StringRef moduleId,
                       llvm::StringRef targetCPU,
                       llvm::raw_ostream &diagOS) {
    if (!tempFiles.create("ptoas-host-stub", ".o", hostStubObjPath, diagOS))
      return false;
    return compileHostStubToFatobj(stubPath, hostStubObjPath, moduleId,
                                   targetCPU, toolchain, mergedDeviceObjPath,
                                   stderrPath, diagOS);
  }

  bool repackFatObj(const VPTOFatobjToolchain &toolchain,
                    llvm::StringRef moduleId, llvm::StringRef targetCPU,
                    llvm::StringRef outPath, llvm::raw_ostream &diagOS) {
    llvm::SmallVector<std::string, 16> args = {
        toolchain.cceLd(),
        toolchain.ldLld(),
        "-x",
        "-cce-lite-bin-module-id",
        moduleId.str(),
        std::string("-cce-aicore-arch=") + targetCPU.str(),
        "-r",
        "-o",
        outPath.str(),
        "-cce-stub-dir",
        toolchain.cceStubDir(),
        "-cce-install-dir",
        toolchain.bishengCompilerBinDir(),
        "-cce-inputs-number",
        "1",
        hostStubObjPath,
    };
    return runCommandWithStderr(toolchain.cceLd(), args, stderrPath, diagOS,
                                "fatobj repack");
  }

private:
  TempFileRegistry &tempFiles;
  std::string cubeLLPath;
  std::string cubeObjPath;
  std::string vectorLLPath;
  std::string vectorObjPath;
  std::string mergedDeviceObjPath;
  std::string stderrPath;
  std::string stubPath;
  std::string hostStubObjPath;
};

static bool runCommandWithStderr(llvm::StringRef program,
                                 llvm::ArrayRef<std::string> ownedArgs,
                                 llvm::StringRef stderrPath,
                                 llvm::raw_ostream &diagOS,
                                 llvm::StringRef what,
                                 std::optional<llvm::StringRef> stdinPath) {
  llvm::SmallVector<llvm::StringRef, 16> args;
  args.reserve(ownedArgs.size());
  for (const std::string &arg : ownedArgs)
    args.push_back(arg);
  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects = {
      stdinPath, std::nullopt, stderrPath};

  std::string execErr;
  bool execFailed = false;
  int rc = llvm::sys::ExecuteAndWait(program, args, std::nullopt, redirects, 0,
                                     0, &execErr, &execFailed);
  if (!execFailed && rc == 0)
    return true;

  diagOS << "Error: " << what << " failed\n";
  diagOS << "Command:";
  for (llvm::StringRef arg : args)
    diagOS << " " << arg;
  diagOS << "\n";
  if (!execErr.empty())
    diagOS << execErr << "\n";
  if (auto buffer = llvm::MemoryBuffer::getFile(stderrPath))
    diagOS << buffer.get()->getBuffer() << "\n";
  return false;
}

static bool compileDeviceLLVMToObject(llvm::StringRef llPath,
                                      llvm::StringRef outObjPath,
                                      llvm::StringRef targetCPU,
                                      llvm::StringRef bishengPath,
                                      llvm::StringRef stderrPath,
                                      llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, 16> args = {
      bishengPath.str(),
      "--target=hiipu64-hisilicon-cce",
      std::string("-march=") + targetCPU.str(),
      std::string("--cce-aicore-arch=") + targetCPU.str(),
      "--cce-aicore-only",
      "-O2",
      "-c",
      "-x",
      "ir",
      "-",
      "-o",
      outObjPath.str(),
  };
  return runCommandWithStderr(bishengPath, args, stderrPath, diagOS,
                              "device LLVM compilation", llPath);
}

static bool compileHostStubToFatobj(llvm::StringRef stubPath,
                                    llvm::StringRef outObjPath,
                                    llvm::StringRef moduleId,
                                    llvm::StringRef targetCPU,
                                    const VPTOFatobjToolchain &toolchain,
                                    llvm::StringRef deviceObjPath,
                                    llvm::StringRef stderrPath,
                                    llvm::raw_ostream &diagOS) {
  std::string coverageDir = ".";
  std::string debugDir = ".";
  std::string hostTriple = llvm::sys::getProcessTriple();

  llvm::SmallVector<std::string, 32> args = {
      toolchain.bishengCc1(),
      "-cc1",
      "-triple",
      hostTriple,
      "-target-cpu",
      llvm::sys::getHostCPUName().str(),
      "-fcce-aicpu-legacy-launch",
      "-fcce-is-host",
      "-cce-enable-mix",
      "-mllvm",
      "-enable-mix=true",
      "-cce-launch-with-flagv2-impl",
      "-fcce-aicore-arch",
      targetCPU.str(),
      "-fcce-fatobj-compile",
      "-emit-obj",
      "--mrelax-relocations",
      "-disable-free",
      "-clear-ast-before-backend",
      "-disable-llvm-verifier",
      "-discard-value-names",
      "-main-file-name",
      "stub.cpp",
      "-mrelocation-model",
      "pic",
      "-pic-level",
      "2",
      "-fhalf-no-semantic-interposition",
      "-mframe-pointer=none",
      "-fmath-errno",
      "-ffp-contract=on",
      "-fno-rounding-math",
      "-mconstructor-aliases",
      "-funwind-tables=2",
      "-fallow-half-arguments-and-returns",
      "-mllvm",
      "-treat-scalable-fixed-error-as-warning",
      std::string("-fcoverage-compilation-dir=") + coverageDir,
      "-resource-dir",
      toolchain.resourceDir(),
      "-internal-isystem",
      toolchain.resourceIncludeDir(),
      "-include",
      "__clang_cce_runtime_wrapper.h",
      "-D",
      "_FORTIFY_SOURCE=2",
      "-D",
      "REGISTER_BASE",
      "-O2",
      "-Wno-macro-redefined",
      "-Wno-ignored-attributes",
      "-std=c++17",
      "-fdeprecated-macro",
      std::string("-fdebug-compilation-dir=") + debugDir,
      "-ferror-limit",
      "19",
      "-stack-protector",
      "2",
      "-fno-signed-char",
      "-fgnuc-version=4.2.1",
      "-fcxx-exceptions",
      "-fexceptions",
      "-vectorize-loops",
      "-vectorize-slp",
      "-mllvm",
      "-cce-aicore-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-function-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-record-overflow=true",
      "-mllvm",
      "-cce-aicore-addr-transform",
      "-mllvm",
      "-cce-aicore-dcci-insert-for-scalar=false",
      "-fcce-include-aibinary",
      deviceObjPath.str(),
      "-fcce-device-module-id",
      moduleId.str(),
      "-faddrsig",
      "-D__GCC_HAVE_DWARF2_CFI_ASM=1",
      "-o",
      outObjPath.str(),
      "-x",
      "cce",
      stubPath.str(),
  };
  return runCommandWithStderr(toolchain.bishengCc1(), args, stderrPath, diagOS,
                              "host stub compilation");
}

static bool mergeDeviceObjects(llvm::ArrayRef<std::string> deviceObjPaths,
                               llvm::StringRef outObjPath,
                               llvm::StringRef ldLldPath,
                               llvm::StringRef stderrPath,
                               llvm::raw_ostream &diagOS) {
  if (deviceObjPaths.empty())
    return false;

  llvm::SmallVector<std::string, 16> args = {
      ldLldPath.str(),
      "-m",
      "aicorelinux",
      "-Ttext",
      "0",
  };
  for (const std::string &path : deviceObjPaths)
    args.push_back(path);
  args.push_back("-o");
  args.push_back(outObjPath.str());
  args.push_back("-r");
  args.push_back("--allow-multiple-definition");
  return runCommandWithStderr(ldLldPath, args, stderrPath, diagOS,
                              "device object merge");
}

} // namespace

mlir::LogicalResult mlir::pto::emitVPTOFatobj(llvm::Module *cubeModule,
                                              llvm::Module *vectorModule,
                                              llvm::StringRef stubSource,
                                              llvm::ToolOutputFile &outputFile,
                                              llvm::raw_ostream &diagOS,
                                              llvm::StringRef deviceTargetCPU) {
  if (!cubeModule && !vectorModule) {
    diagOS << "Error: VPTO fatobj emission requires at least one LLVM module.\n";
    return failure();
  }

  std::string vectorCPU = deviceTargetCPU.str();
  std::string cubeCPU = deviceTargetCPU.str();
  size_t vecPos = cubeCPU.rfind("-vec");
  if (vecPos != std::string::npos)
    cubeCPU.replace(vecPos, 4, "-cube");

  std::optional<VPTOFatobjToolchain> toolchain =
      VPTOFatobjToolchain::create(diagOS);
  if (!toolchain)
    return failure();

  TempFileRegistry tempFiles;
  VPTOFatobjArtifacts artifacts(tempFiles);
  if (!artifacts.emitStubSource(stubSource, diagOS))
    return failure();
  if (!artifacts.initCommandLogs(diagOS))
    return failure();

  if (!artifacts.emitCubeObject(cubeModule, *toolchain, cubeCPU, diagOS))
    return failure();
  if (!artifacts.emitVectorObject(vectorModule, *toolchain, vectorCPU, diagOS))
    return failure();

  if (!artifacts.mergeDeviceObjects(*toolchain, diagOS))
    return failure();

  std::string moduleId = sanitizeModuleId(outputFile.getFilename());
  constexpr llvm::StringLiteral hostTargetCPU = "dav-c310";
  if (!artifacts.compileHostStub(*toolchain, moduleId, hostTargetCPU, diagOS))
    return failure();

  if (!artifacts.repackFatObj(*toolchain, moduleId, hostTargetCPU,
                              outputFile.getFilename(), diagOS))
    return failure();
  outputFile.keep();
  return success();
}
