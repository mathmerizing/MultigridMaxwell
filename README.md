# MultigridMaxwell

## 1. Overview
This code belongs to my bachelor's thesis and it uses [deal.ii](https://github.com/dealii/dealii) to solve the Eddy Current problem.

## 2. Installation
Clone this repository, install cmake and a version of deal.ii which contains [my merged Pull Request](https://github.com/dealii/dealii/pull/10348#event-3421712455).

## 3. Change deal.II path
Change `DEAL_II_DIR` in `CMakeLists.txt` to your path to the deal.ii library.

## 4. Run
Navigate to this local GitHub repository in your terminal and execute

`make clean;rm -r CMakeFiles/;cmake .;make release;make -j 8;./EddyCurrent`

## 5. OPTIONAL: Visualization
The file `solution.vtu` can be visualized with [ParaView](https://www.paraview.org/).
