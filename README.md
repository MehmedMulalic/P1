# Praktikum P1 project for University of Vienna Faculty of Computer Science

This project aims to showcase the difference in performance between three different heterogeneous parallel framworks: Mojo, SYCL, and OpenCL.

---

## Installation

After cloning the repository, run the install script which installs all requirements and builds the project.

```bash
git clone https://github.com/MehmedMulalic/P1.git ~/P1
chmod +x ~/P1/install.sh
./install.sh
```

---

## Usage

There are five executables which have the same argument parsing usage:  
Usage (positive particle_count, positive grid_size): <particle_count>, <grid_size>

The default values for atom count and grid mesh size is (1, 1) but they can be changed if parsed with the described usage.
