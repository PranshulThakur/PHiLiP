Include "airfoil_naca0012.geo";

//+
ymax = 10;
xmax = 10;
refinement_level = 0;
n_inlet = 9;
n_vertical = 18;
r_vertical = 1.5;
n_airfoil = 8;
n_wake = 12;
//r_wake = 1/0.93;
r_wake = 1.5;

//+
Point(131) = {-0.5, ymax, 0, 1.0};
//+
Point(132) = {-0.5, -ymax, 0, 1.0};
//+
Point(133) = {1, ymax, 0, 1.0};
//+
Point(134) = {1, -ymax, 0, 1.0};
//+
Point(135) = {xmax, ymax, 0, 1.0};
//+
Point(136) = {xmax, -ymax, 0, 1.0};
//+
Point(137) = {xmax, 0, 0, 1.0};

//+
Circle(2) = {132, 65, 131};
//+
Line(3) = {57, 131};
//+
Line(4) = {73, 132};
//+
Line(5) = {131, 133};
//+
Line(6) = {132, 134};
//+
Line(7) = {133, 135};
//+
Line(8) = {134, 136};
//+
Line(9) = {137, 136};
//+
Line(10) = {137, 135};
//+
Line(11) = {130, 133};
//+
Line(12) = {130, 134};
//+
Line(13) = {130, 137};
//+
Split Curve(1) {57, 73};
//+
Split Curve(15) {130};
//+
Transfinite Curve {2, 14} = n_inlet Using Progression 1;
//+
Transfinite Curve {3, 11, 10, 4, 12, 9} = n_vertical Using Progression r_vertical;
//+
Transfinite Curve {17, 16} = n_airfoil Using Bump 0.1;
//+
Transfinite Curve {5 , 6} = n_airfoil Using Bump 2;
//+
Transfinite Curve {13} = n_wake Using Progression r_wake;
//+
//Transfinite Curve {7, 8} = n_wake Using Bump 0.2;
Transfinite Curve {7, 8} = n_wake Using Progression r_wake;
//+
Curve Loop(1) = {2, -3, 14, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {3, 5, -11, 17};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {11, 7, -10, -13};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {16, 12, -6, -4};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {13, 9, -8, -12};
//+
Plane Surface(5) = {5};
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Transfinite Surface {3};
//+
Transfinite Surface {5};
//+
Transfinite Surface {4};
//+
Recombine Surface {1, 2, 3, 5, 4};
//+
Physical Curve("Farfield", 1004) = {2, 6, 8, 9, 10, 7, 5};
//+
Physical Surface("MeshInterior") = {1, 2, 3, 4, 5};
//+
Physical Curve("Airfoil", 1001) = {17, 14, 16};

// Use a specific 2D algorithm (e.g., 8 = Delaunay for Quads, 11 = Quasi-structured Quads)
Mesh.Algorithm = 8;

// Set recombination strategy (1 = Blossom algorithm, isolates quads cleanly)
Mesh.RecombineAll = 1;

Mesh 2;
SetOrder 2;
For i In {1:refinement_level}
    RefineMesh;
    SetOrder 2;
EndFor
//+
Show "*";
//+
Show "*";
