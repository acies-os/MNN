[private]
default:
    @just -f {{ justfile() }} --list

# if os() == "macos" and arch() == "aarch64", set `mac-m1-flag` to "-DCMAKE_OSX_ARCHITECTURES=arm64" else ""
# mac-m1-flag := if os() == 'macos' && arch() == 'aarch64' { "-DCMAKE_OSX_ARCHITECTURES=arm64" } else { "" }

mac-m1-flag := if os() == "macos" { if arch() == "aarch64" { "-DCMAKE_OSX_ARCHITECTURES=arm64" } else { "" } } else { "" }
flags := "-DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1"

# Setup a bare repository and add remotes for MNN (upstream) and Melon
setup:
    git remote add upstream https://github.com/alibaba/MNN.git
    git remote add melon https://github.com/acies-os/Melon.git
    git fetch upstream
    git fetch melon
    git worktree add ../melon-main melon/main
    git worktree add -b acies/melon-base ../melon-base 6a6d1a095d0cd573bb67ba9608e6c686ca11c834

build n:
    ./schema/generate.sh
    ./tools/script/get_model.sh
    mkdir -p build && cd build && cmake .. {{ mac-m1-flag }} {{ flags }} && make -j{{ n }}

echo:
    echo {{ mac-m1-flag }}
