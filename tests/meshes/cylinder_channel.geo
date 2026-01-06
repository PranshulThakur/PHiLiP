//+
r1 = 0.5;
r2 = 30.0;
theta1 = 0;//Pi/4;
theta2 = Pi/2;//3*Pi/4;
theta3 = Pi;//5*Pi/4;
theta4 = 3*Pi/2;//7*Pi/4;
Point(1) = {0, 0, 0, 1.0};
Point(2) = {r1*Cos(theta1), r1*Sin(theta1), 0, 1.0};
Point(3) = {r1*Cos(theta2), r1*Sin(theta2), 0, 1.0};
Point(4) = {r1*Cos(theta3), r1*Sin(theta3), 0, 1.0};
Point(5) = {r1*Cos(theta4), r1*Sin(theta4), 0, 1.0};
Point(6) = {r2*Cos(theta1), r2*Sin(theta1), 0, 1.0};
Point(7) = {r2*Cos(theta2), r2*Sin(theta2), 0, 1.0};
Point(8) = {r2*Cos(theta3), r2*Sin(theta3), 0, 1.0};
Point(9) = {r2*Cos(theta4), r2*Sin(theta4), 0, 1.0};
//+
Circle(1) = {4, 1, 5};
//+
Circle(2) = {5, 1, 2};
//+
Circle(3) = {2, 1, 3};
//+
Circle(4) = {3, 1, 4};
//+
Circle(5) = {8, 1, 9};
//+
Circle(6) = {9, 1, 6};
//+
Circle(7) = {6, 1, 7};
//+
Circle(8) = {7, 1, 8};
//+
Line(9) = {9, 5};
//+
Line(10) = {4, 8};
//+
Line(11) = {3, 7};
//+
Line(12) = {2, 6};
//+
Curve Loop(1) = {10, 5, 9, -1};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {9, 2, 12, -6};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {12, 7, -11, -3};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {11, 8, -10, -4};
//+
Plane Surface(4) = {4};
//+
Transfinite Curve {9, 12, 11, 10} = 10 Using Progression 1;
//+
Transfinite Curve {6, 7, 8, 5, 1, 2, 3, 4} = 10 Using Progression 1;
//+
Extrude {0, 0, 2} {
  Curve{6}; Curve{7}; Curve{8}; Curve{5}; Curve{9}; Curve{12}; Curve{11}; Curve{10}; Curve{3}; Curve{4}; Curve{1}; Curve{2}; Surface{2}; Surface{3}; Surface{4}; Surface{1}; Layers {3}; Recombine;
}
