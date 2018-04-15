(ns main
  (:import [org.tensorflow DataType Graph Output Session Tensor TensorFlow]
           [org.tensorflow.types.UInt8]))

(defn -main []
  (prn :imported (. TensorFlow version)))
