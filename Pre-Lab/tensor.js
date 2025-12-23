const tensorA = tf.tensor([1, 2]);
console.log("Tensor A:");
tensorA.print();

const tensorB = tf.tensor([3, 4]);
console.log("Tensor B:");
tensorB.print();

const result = tensorA.add(tensorB);
console.log("Result (Tensor A + Tensor B):");
result.print();