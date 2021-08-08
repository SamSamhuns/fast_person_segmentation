import os
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model


# clear prev session
tf.keras.backend.clear_session()


def freeze_graph(graph, session, output, save_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(
            graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(
            session, graphdef_inf, output)

        # To add an init operation to the model
        init_op = tf.initializers.global_variables()
        graph_io.write_graph(graphdef_frozen, save_dir,
                             save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen


def save_pb(model_path, save_pb_dir, save_pb_name):
    # This line must be executed before loading Keras model
    tf.keras.backend.set_learning_phase(0)
    model = load_model(model_path, compile=True)
    session = tf.keras.backend.get_session()

    INPUT_NODE = [t.op.name for t in model.inputs]
    OUTPUT_NODE = [t.op.name for t in model.outputs]
    print(INPUT_NODE, OUTPUT_NODE)

    os.makedirs(save_pb_dir, exist_ok=True)
    freeze_graph(session.graph,
                 session,
                 [out.op.name for out in model.outputs],
                 save_dir=save_pb_dir,
                 save_pb_name=save_pb_name)


def main(model_path, save_pb_dir, save_pb_name):
    save_pb(model_path, save_pb_dir, save_pb_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dump hdf5 model to pb file')
    parser.add_argument('-m', '--model_path', type=str,
                        help='path to hdf5/h5 model', default="")
    parser.add_argument('-sp', '--save_pb_dir', type=str,
                        help='folder path to save pb file', default="models")
    parser.add_argument('-p', '--save_pb_name', type=str,
                        help='pb file name', default="frozen_model.pb")
    args = parser.parse_args()

    main(args.model_path,
         args.save_pb_dir,
         args.save_pb_name)
