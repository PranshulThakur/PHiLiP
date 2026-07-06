Include "airfoil_naca0012.geo";
Split Curve(1) {57, 73};
Split Curve(3) {130};
// 2. Define Farfield
lc_farfield = 4.0;
farfield_distance = 10;
//Point(131) = {0.5, 0.0, 0.0};
Point(132) = {-farfield_distance+0.0,0.0,0.0,lc_farfield};
Point(133) = {0.0,farfield_distance,0.0,lc_farfield};
Point(134) = {farfield_distance+0.0,0.0,0.0,lc_farfield};
Point(135) = {0.0,-farfield_distance,0.0,lc_farfield};

Circle(141) = {132,65,133};
Circle(142) = {133,65,134};
Circle(144) = {134,65,135};
Circle(145) = {135,65,132};


// Create a closed loop for the airfoil and the farfield
Curve Loop(1) = {2,4,5}; //Airfoil
Curve Loop(2) = {141, 142, 144, 145}; //Farfield

// 3. Create Surface and Mesh
Plane Surface(1) = {2, 1};
Recombine Surface{1};
// 4. Assign Physical Groups (Boundary IDs)
Physical Curve(1001) = {2,4,5}; // Airfoil Surface
Physical Curve(1004) = {141, 142, 144, 145}; // Farfield Boundary
Physical Surface(10) = {1}; // Fluid Volume




Mesh.HighOrderCurveOuterBL = 1;   // Default: 0 (0 = do not curve, 1 = curve according to boundary, …)
Mesh.HighOrderFastCurvingNewAlgo = 1;
Mesh.HighOrderOptimize = 4;       // Default: 0 (0: none, …, 4: fast curving)
Mesh.SecondOrderLinear = 0;

// Structured grid around airfoil
y_plus = 0.02;
Field[2] = BoundaryLayer;
Field[2].CurvesList = {2,4,5};
Field[2].SizeFar = lc_farfield;
Field[2].Size = y_plus;
Field[2].Thickness = 0.05;
Field[2].Ratio = 1.1;
Field[2].FanPointsList = {130};
BoundaryLayer Field = 2;

Field[8] = Cylinder;
Field[8].Radius = 0.05;
Field[8].XCenter = 0.0;
Field[8].YCenter = 0.0;//-0.00186*Scaling;
Field[8].VIn = 0.02;//0.025;
Field[8].VOut = lc_farfield; //Same as  farfield

Field[3] = Min;
Field[3].FieldsList = {1,2,8};
Background Field = 3;

Field[1] = Box;
Field[1].VIn = 0.06;
Field[1].VOut = lc_farfield; //Same as  farfield
Field[1].XMax = 3.0;
Field[1].XMin = -0.5;
Field[1].YMax = 1.5;
Field[1].YMin = -0.5;

// Use a specific 2D algorithm (e.g., 8 = Delaunay for Quads, 11 = Quasi-structured Quads)
Mesh.Algorithm = 5;
Mesh.NumSubEdges = 3;
Coherence;

// Set recombination strategy (1 = Blossom algorithm, isolates quads cleanly)
Mesh.RecombineAll = 1;
Mesh 2;
SetOrder 2;


