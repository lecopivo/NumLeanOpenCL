import Lake

open System Lake DSL

package NumLeanOpenCL where
  version := v!"0.1.0"

require "lecopivo" / NumLean from git "https://github.com/lecopivo/NumLean"

target liblean_numleanopencl_opencl pkg : Dynlib := do
  let libName := "lean_numleanopencl_opencl"
  let entries ← (pkg.dir / "c").readDir
  let cFiles := entries.filterMap fun entry =>
    if entry.path.extension == some "c" then
      some entry.path
    else
      none
  let objJobs ← cFiles.mapM fun cFile => do
    let srcJob ← inputFile cFile true
    let oFile := (pkg.buildDir / "c" / cFile.fileName.get!).withExtension "o"
    buildO oFile srcJob #["-I", (← getLeanIncludeDir).toString] #["-fPIC"] "cc" getLeanTrace
  let weakArgs := #[
    "-L", (← getLeanLibDir).toString,
    s!"-Wl,-rpath,{(← getLeanLibDir).toString}",
    "/usr/lib/x86_64-linux-gnu/libOpenCL.so"
  ]
  let leanArgs := (← getLeanLinkSharedFlags).filter (· != "-fuse-ld=lld")
  buildSharedLib libName (pkg.sharedLibDir / nameToSharedLib libName)
    objJobs #[] weakArgs leanArgs "cc" getLeanTrace

lean_lib NumLeanOpenCL where

lean_lib NumLeanOpenCL.OpenCL.Basic where
  precompileModules := true
  moreLinkLibs := #[liblean_numleanopencl_opencl]

lean_lib NumLeanOpenCL.Data.Float32Array.Basic where
  precompileModules := true
  moreLinkLibs := #[liblean_numleanopencl_opencl]

@[default_target] lean_exe numleanopencl where root := `Main

lean_exe opencl_profile where root := `Profile

@[test_driver] lean_exe numleanopencl_tests where
  root := `Tests.Main
  supportInterpreter := true

lean_lib Tests.Eval where
  precompileModules := true
