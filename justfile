[private]
default:
    @just -f {{ justfile() }} --list

# Setup a bare repository and add remotes for MNN (upstream) and Melon
setup:
    git remote add upstream https://github.com/alibaba/MNN.git
    git remote add melon https://github.com/acies-os/Melon.git
    git fetch upstream
    git fetch melon
    git worktree add ../melon-main melon/main
    git worktree add -b acies/melon-base ../melon-base 6a6d1a095d0cd573bb67ba9608e6c686ca11c834
