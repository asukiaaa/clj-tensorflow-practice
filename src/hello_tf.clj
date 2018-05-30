(ns hello-tf
  (:import [org.tensorflow Graph Session Tensor TensorFlow]))

(defn -main []
  (let [g (new Graph)
        value (str "Hello from " (TensorFlow/version))
        t (Tensor/create (byte-array (map byte value)))
        _ (-> (.opBuilder g "Const" "MyConst")
              (.setAttr "dtype" (.dataType t))
              (.setAttr "value" t)
              .build)
        output (-> (new Session g)
                .runner
                (.fetch "MyConst")
                .run
                (.get 0))]
    (prn (String. (.bytesValue output)))))
