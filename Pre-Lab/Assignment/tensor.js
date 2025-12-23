/***********************
 STEP 1: Tensor Creation
************************/
const scalar = tf.scalar(10);
const vector = tf.tensor1d([1, 2, 3]);
const matrix = tf.tensor2d([[1, 2], [3, 4]]);

document.getElementById("step1").innerText =
`Scalar (0-D): ${scalar.toString()}

Vector (1-D): ${vector.toString()}

Matrix (2-D): ${matrix.toString()}`;


/*********************************
 STEP 2: Vector Operations
*********************************/
const vectorA = tf.tensor1d([1, 2, 3]);
const vectorB = tf.tensor1d([4, 5, 6]);

const addition = vectorA.add(vectorB);
const multiplication = vectorA.mul(vectorB);

document.getElementById("step2").innerText =
`Vector A: ${vectorA.toString()}
Vector B: ${vectorB.toString()}

Addition Result: ${addition.toString()}
Multiplication Result: ${multiplication.toString()}`;


/*********************************
 STEP 3: Reshape vs Flatten
*********************************/
const originalTensor = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
const reshapedTensor = originalTensor.reshape([3, 2]);
const flattenedTensor = originalTensor.flatten();

document.getElementById("step3").innerText =
`Original Tensor (2x3):
${originalTensor.toString()}

Reshaped Tensor (3x2):
${reshapedTensor.toString()}

Flattened Tensor (1-D):
${flattenedTensor.toString()}`;
