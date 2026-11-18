// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

// SPDX-License-Identifier: GPL-3.0-or-later

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/sonowing.ico");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let ico_path = manifest_dir.join("assets").join("sonowing.ico");
    if !ico_path.exists() {
        println!(
            "cargo:warning=SonoWing Windows icon not found at {}",
            ico_path.display()
        );
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let rc_path = out_dir.join("app_res.rc");
    let ico_rc = ico_path.to_string_lossy().replace('\\', "/");

    let rc_body = format!(
        r#"1 ICON "{ico_rc}"

1 VERSIONINFO
FILEVERSION 1,1,0,0
PRODUCTVERSION 1,1,0,0
FILEFLAGSMASK 0x3fL
FILEFLAGS 0x0L
FILEOS 0x40004L
FILETYPE 0x1L
FILESUBTYPE 0x0L
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "CompanyName", "CRV Science Network\0"
      VALUE "FileDescription", "SonoWing - Bat & Bird Bioacoustic Workstation\0"
      VALUE "FileVersion", "1.1.0.0\0"
      VALUE "InternalName", "sonowing\0"
      VALUE "LegalCopyright", "Copyright CRV Science Network (GPL-3.0-or-later)\0"
      VALUE "OriginalFilename", "sonowing.exe\0"
      VALUE "ProductName", "SonoWing\0"
      VALUE "ProductVersion", "1.1.0\0"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x0409, 1200
  END
END
"#
    );

    if let Err(e) = fs::write(&rc_path, rc_body) {
        println!("cargo:warning=Failed writing {}: {e}", rc_path.display());
        return;
    }

    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_env == "gnu" {
        let obj_path = out_dir.join("app_res.o");
        let windres_candidates = ["windres", "x86_64-w64-mingw32-windres"];
        for tool in windres_candidates {
            if let Ok(status) = Command::new(tool)
                .arg(&rc_path)
                .arg("-O")
                .arg("coff")
                .arg("-o")
                .arg(&obj_path)
                .status()
            {
                if status.success() {
                    println!("cargo:rustc-link-arg-bins={}", obj_path.display());
                    return;
                }
            }
        }
        println!("cargo:warning=windres not available; skipping Windows PE icon/VersionInfo embedding");
    } else if target_env == "msvc" {
        let res_path = out_dir.join("app_res.res");
        if let Some(rc_exe) = find_rc_exe() {
            if let Ok(status) = Command::new(rc_exe)
                .arg("/nologo")
                .arg("/fo")
                .arg(&res_path)
                .arg(&rc_path)
                .status()
            {
                if status.success() {
                    println!("cargo:rustc-link-arg-bins={}", res_path.display());
                    return;
                }
            }
        }
        println!("cargo:warning=rc.exe not available; skipping Windows PE icon/VersionInfo embedding");
    }
}

fn find_rc_exe() -> Option<PathBuf> {
    if Command::new("rc.exe").arg("/?").output().is_ok() {
        return Some(PathBuf::from("rc.exe"));
    }
    let kits_root = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    if !kits_root.exists() {
        return None;
    }
    let mut candidates: Vec<PathBuf> = fs::read_dir(kits_root)
        .ok()?
        .filter_map(|e| e.ok().map(|d| d.path().join("x64").join("rc.exe")))
        .filter(|p| p.exists())
        .collect();
    candidates.sort();
    candidates.pop()
}
