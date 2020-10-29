with (import <nixpkgs> {});

stdenv.mkDerivation rec {
  name = "760-smeter-build-env";
  env = buildEnv { name = name; paths = buildInputs; };
  buildInputs =
    let pyenv = python37.withPackages (pps: with pps;

      [
        scikitlearn
        tensorflow
        numpy
        flake8
        black
      ]
    );
    in [ pyenv gnumake ];
}
