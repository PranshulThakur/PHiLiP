r1 = 1;
r2 = 1.384;
Point(1) = {0,0,0,1};
Point(2) = {r1,0,0,1};
Point(3) = {r2,0,0,1};
Point(4) = {0,r1,0,1};
Point(5) = {0,r2,0,1};
//+
Circle(1) = {4, 1, 2};
//+
Circle(2) = {5, 1, 3};
//+
Line(3) = {5, 4};
//+
Line(4) = {3, 2};
//+
Curve Loop(1) = {3, 1, -4, -2};
//+
Plane Surface(1) = {1};
//+
Physical Surface("innervol", 5) = {1};
//+
Physical Curve("slipwall", 1001) = {2};
//+
Physical Curve("riemanninvariant", 1004) = {3, 4};
//+
Physical Curve("slipwallinflow", 1009) = {1};
//+
Transfinite Curve {1, 2} = 31 Using Progression 1;
//+
Transfinite Curve {3, 4} = 6 Using Progression 1;
//+
Transfinite Surface {1};

Mesh.RecombineAll = 1;
//+
Mesh.RecombinationAlgorithm = 2;
