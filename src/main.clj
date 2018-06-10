(ns main
  (:import [java.lang.String]
           [java.io File FileInputStream]
           [org.tensorflow DataType Graph Output Session Tensor TensorFlow]
           [org.tensorflow.types.UInt8])
  (:require [clojure.java.io :as io]))

(def graph-pb-path "data/tensorflow_inception_graph.pb")
(def graph-labels-path "data/imagenet_comp_graph_label_strings.txt")
(def image-path "img/lena.jpg")

(defn load-byte-array [file-path]
  (let [f (File. file-path)
        array (byte-array (.length f))
        stream (FileInputStream. f)]
    (.read stream array)
    (.close stream)
    array))

(defn load-labels [file-path]
  (with-open [file-reader (io/reader file-path)]
    (doall (line-seq file-reader))))

(defn get-constant
  ([g name value type]
   (prn name value type)
   (let [t (Tensor/create value)]
     (-> (.opBuilder g "Const" name)
         (.setAttr "dtype" (DataType/fromClass type))
         (.setAttr "value" t)
         .build
         (.output 0))))
  ([g name value]
   (cond (integer? value) (get-constant g name (Integer. value) Integer)
         (float? value) (get-constant g name (Float. value) Float)
         (and (vector? value) (integer? (first value))) (get-constant g name (int-array value) Integer)
         :else (get-constant g name value String))))

(defn decode-jpeg [g contents channels]
  (-> (.opBuilder g "DecodeJpeg" "DecodeJpeg")
      (.addInput contents)
      (.setAttr "channels" channels)
      .build
      (.output 0)))

(defn get-cast [g value type]
  (let [dtype (DataType/fromClass type)]
    (-> (.opBuilder g "Cast" "Cast")
        (.addInput value)
        (.setAttr "DstT" dtype)
        .build
        (.output 0))))

(defn get-binary-op [g type in1 in2]
  (-> (.opBuilder g type type)
      (.addInput in1)
      (.addInput in2)
      .build
      (.output 0)))

(defn get-div [g x y]
  (get-binary-op g "Div" x y))

(defn get-sub [g x y]
  (get-binary-op g "Sub" x y))

(defn get-binary-op3 [g type in1 in2]
  (-> (.opBuilder g type type)
      (.addInput in1)
      (.addInput in2)
      .build
      (.output 0)))

(defn expand-dims [g input dim]
  (get-binary-op3 g "ExpandDims" input dim))

(defn resize-bilinear [g images size]
  (get-binary-op3 g "ResizeBilinear" images size))

(defn image-bytes->normalized-image [image-bytes]
  (let [g (new Graph)
        h 224
        w 224
        mean 117.0
        scale 1.0
        input-str (get-constant g "input" image-bytes)
        output (get-div g
                        (get-sub g
                                 (resize-bilinear g
                                                  (expand-dims g
                                                               (get-cast g (decode-jpeg g input-str 3) Float)
                                                               (get-constant g "make_batch" 0))
                                                  (get-constant g "size" [w h]))
                                 (get-constant g "mean" mean))
                        (get-constant g "scale" scale))]
    (-> (new Session g)
        .runner
        (.fetch (-> output
                    .op
                    .name))
        .run
        (.get 0)
        (.expect Float))))

(defn get-label-probabilities [graph-pb image]
  (let [g (new Graph)
        _ (.importGraphDef g graph-pb)
        result (-> (new Session g)
                   .runner
                   (.feed "input" image)
                   (.fetch "output")
                   .run
                   (.get 0)
                   (.expect Float))
        rshape (.shape result)]
    ;(prn :nth0 (nth rshape 0))
    ;(prn :nth1 (nth rshape 1))
    #_(when (or (not= 2 (.numDimensions result)) (not= 1 (nth rshape 0)))
      (throw (RuntimeException "unexpected dimension or shape")))
    (let [nlabels (nth rshape 1)
          array-result (make-array Float/TYPE 1 nlabels)]
      (.copyTo result array-result)
      (nth array-result 0))))

(defn -main []
  (prn :imported-tensorflow-version (. TensorFlow version))
  (let [graph-pb (load-byte-array graph-pb-path)
        labels (load-labels graph-labels-path)
        image-bytes (load-byte-array image-path)
        normalized-image (image-bytes->normalized-image image-bytes)
        label-probabilities (get-label-probabilities graph-pb normalized-image)
        sorted-probabilities (reverse (sort-by second (map-indexed vector label-probabilities)))]
    (prn label-probabilities)
    (prn :graph-byte-size (count graph-pb))
    (prn :label-count (count labels))
    (prn :last-label (last labels))
    (prn :top-labels)
    (doseq [i (range 5)
            :let [p (nth sorted-probabilities i)
                  label (nth labels (first p))
                  prob (second p)]]
      (prn label prob))))
