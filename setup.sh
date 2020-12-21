#!/bin/sh

macOS=false

check_requirements() {
  case "$(uname -s)" in
    Darwin)
      printf 'Installing on macOS'
      export CFLAGS='-stdlib=libc++'
      macOS=true
      ;;
    Linux)
      printf 'Installing on Linux'
      ;;
    *)
      echo 'Only Linux and macOS are currently supported.\n'
      exit 1
      ;;
  esac
}

install_packages() {
  printf "\nInstalling tensorflow..\n"
  if [ $macOS = "true" ]; then
    pip install -q tensorflow==2.4.0
  else
    conda install -q -c nvidia cudatoolkit=11.0 cudnn=8.0 nccl -y
    pip install -q tensorflow==2.4.0
  fi
  printf "\nInstalling other Python packages..."
  pip install -q -r requirements.txt
}

check_requirements
install_packages

printf '\nSetup completed.\n'
