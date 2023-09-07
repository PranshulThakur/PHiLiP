Point(1) = {0 ,0 ,0 , 1.0};
Point(2) = {0 ,1 ,0 , 1.0};
Point(3) = {1 ,0 ,0 , 1.0};
Point(4) = {1 ,1 ,0 , 1.0};

shock_left = 3.0/8.0;
Point(5) = {shock_left, 1, 0, 1.0};
Point(6) = {shock_left, 0, 0, 1.0};

shock_right = 0.6;
Point(7) = {shock_right, 1, 0, 1.0};
Point(8) = {shock_right, 0, 0, 1.0};

//+
Line(1) = {1, 6};
//+
Line(2) = {6, 8};
//+
Line(3) = {8, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 7};
//+
Line(6) = {7, 5};
//+
Line(7) = {5, 2};
//+
Line(8) = {2, 1};
//+
Line(9) = {6, 5};
//+
Line(10) = {8, 7};
//+
Curve Loop(1) = {1, 9, 7, 8};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, 10, 6, -9};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {3, 4, 5, -10};
//+
Plane Surface(3) = {3};
//+
Physical Surface("innervol", 11) = {1, 2, 3};
//+
Physical Curve("one", 1) = {4};
//+
Physical Curve("two", 2) = {1, 2, 3};
//+
Physical Curve("six", 6) = {8};
//+
Physical Curve("three", 3) = {7, 6, 5};

n_vertical = 9;
progression_vertical = 1;

n_left = 3;
progression_left = 1;
n_mid = 2;
progression_mid = 1;
n_right = 3;
progression_right = 1;

//+
Transfinite Curve {8, 9, 10, 4} = n_vertical Using Progression progression_vertical;  // vertical lines
//+
Transfinite Curve {1, 7} = n_left Using Progression progression_left; // horizontal left
//+
Transfinite Curve {2, 6} = n_mid Using Progression progression_mid; // horizontal mid
//+
Transfinite Curve {3, 5} = n_right Using Progression progression_right; // horizontal right
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Transfinite Surface {3};

//+
Mesh.RecombineAll = 1;
//+
Mesh.RecombinationAlgorithm = 2;

Color Red{Surface{1};}
Color Red{Surface{2};}
Color Red{Surface{3};}
