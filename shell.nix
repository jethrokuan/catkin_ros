# { nix-ros ? import (builtins.fetchTarball {
#   name = "nix-ros-overlay";
#   url = "https://github.com/lopsided98/nix-ros-overlay/archive/master.tar.gz";
# })}:
{ nix-ros ? import <nix-ros> }:
let pkgs = nix-ros {
  overlays = [(import ./overlay.nix)];
};
in
with pkgs;
with rosPackages.melodicPython3;
mkShell {
  buildInputs = [
    glibcLocales
    (buildEnv { paths = [
                  moveit
                  # franka-ros
    ]; })
  ];
}
