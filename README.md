# Raytracing
This algorithm can rendering a picture or a set of pictures which can perform the following tasks:
starts with user-defined plane and balls
<br> User-defined world construction: user may decide where the light point is, the color of the light, and the intensity of the ambient light.
<br> User-defined object: user can decide the position and other properties of plane and balls in the world, such as color, optical properties etc. User may also define number and radius of balls.
<br> Rendering emulational pictures from the angle user want to capture the virtual world. (user-defined camera position and direction)
<br> 
<br> Input Requirements：
<br> To run this algorithm, the input should be:
<br> - A set of array defining the position, normal vector, and RGB color of balls, the plane, and light. another set of array defning the position and direction of the camera.
<br> - Floats defining the ambient light intensity, radius of the ball and optical properties of each ball and plane (there's default value for optical properties）.
<br> 
<br> Output:
<br> - an image showing the virtual world.
<br> 
<br> How to use:
<br> 1. User may change the parameters of the world in "World construction" part in main function.
<br> 2. There's four example code in main function. One is image sequence rendering, and three are image sequence rendering (1 light rotation, 2 camera movement, 3 camera rotation), they will save the set of image in folder 'sequence'. User may try them seperately.