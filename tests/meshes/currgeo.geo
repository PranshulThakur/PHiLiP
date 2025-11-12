// file:///home/pranshul/Libraries/dealii/install/doc/doxygen/deal.II/namespaceGridGenerator.html#add14cab546d033c1eaacc9234c64ebcd
// Use plate_with_a_hole()

// Use the OpenCASCADE kernel for robust 3D geometry operations
SetFactory("OpenCASCADE");

// Define parameters for the geometry
r = 0.5;   // Cylinder radius
H = 2.0;   // Cylinder height
L = 10.0;  // Domain length (X-dir)
W = 4.0;   // Domain width (Y-dir)
D = 4.0;   // Domain height (Z-dir)

// Define mesh element size (adjust as needed)
lc = 0.5;

// 1. Create the outer box (external domain)
Box(1) = {0, -W/2, -D/2, L, W, D};

// 2. Create the inner cylinder, centered at (L/4, 0, 0)
Cylinder(2) = {L/4, 0, 0, 0, 0, H, r}; // Base at (L/4, 0, 0), direction (0,0,H)

// 3. Cut the cylinder from the box to create the fluid volume
// The 'cut' operation defines the final meshing volume
Volume(3) = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };

// Synchronize the OCC entities with the Gmsh model
Synchronize;

// 4. Define mesh properties for hex meshing
// Set global mesh size
Mesh.CharacteristicLengthMax = lc;

// Enable mesh recombination for quadrilateral/hexahedral elements
Mesh.Algorithm = 1; // 1 = MeshAdapt, 2 = Automatic, 5 = Delaunay, 6 = Frontal, 7 = BAMG, 8 = Netgen
Mesh.Algorithm3D = 4; // 4 = Frontal, 7 = Netgen, 9 = R-tree
Mesh.RecombineAll = 1; // Recombine triangles into quads, tets into hexes if possible

// Use a size field for finer mesh near the cylinder (optional but recommended for flow sims)
Field[1] = Distance;
Field[1].FacesList = {Volume{2}}; // This will list the original cylinder surfaces
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc / 5;   // Minimum element size near cylinder
Field[2].LcMax = lc;      // Maximum element size away from cylinder
Field[2].DistMin = r * 1.1; // Distance from cylinder for min size
Field[2].DistMax = W;       // Distance from cylinder for max size
Mesh.CharacteristicLengthField = 2;

// 5. Define Physical Groups for boundary conditions (essential for simulations)

// Get all surfaces of the resulting fluid volume
surfaces[] = Geometry.Surface { Volume{3} };

// Define inlet, outlet, walls, and cylinder surface
// (Assuming standard CFD orientation: x-inflow, x-outflow, y/z walls, cylinder wall)

// Inlet (x=0 surface)
Physical Surface("Inlet") = {surfaces[0]}; // Index will vary, check in GUI
// Outlet (x=L surface)
Physical Surface("Outlet") = {surfaces[1]};
// Bottom wall (z=-D/2)
Physical Surface("Bottom") = {surfaces[2]};
// Top wall (z=D/2)
Physical Surface("Top") = {surfaces[3]};
// Back wall (y=-W/2)
Physical Surface("Back") = {surfaces[4]};
// Front wall (y=W/2)
Physical Surface("Front") = {surfaces[5]};
// Cylinder wall (inner surface)
Physical Surface("Cylinder_Wall") = {surfaces[6]};

// Define the fluid volume
Physical Volume("Fluid") = {3};

// 6. Generate the 3D mesh
Mesh 3;

// Save the mesh file (e.g., in MSH format, version 2 recommended for compatibility)
GmshWrite "flow_domain.msh";
