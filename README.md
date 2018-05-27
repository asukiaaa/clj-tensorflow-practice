# clj-tensorflow-practice

# Tested environment

Name | Version
---- | ----
OS   | Ubuntu 17.10
Java | open-jdk-1.8
clj  | 1.9.0
CPU  | Intel Core i7 4600U

# Download pretrained data

```
mkdir data
cd data
wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
unzip inception5h.zip
cd ../
```

# Usage

```
clj -m main
```

# License

MIT

# References

- [kieranbrowne/clojure-tensorflow-interop](https://github.com/kieranbrowne/clojure-tensorflow-interop)
- [Installing TensorFlow for Java](https://www.tensorflow.org/install/install_java)
- [TensorFlow for Java](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/README.md)
- [Maven Repository: org.tensorflow » tensorflow](https://mvnrepository.com/artifact/org.tensorflow/tensorflow)
- [tensorflow/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java)
- [Stuck with type hints in clojure for generic class](https://stackoverflow.com/questions/32122495/stuck-with-type-hints-in-clojure-for-generic-class)
