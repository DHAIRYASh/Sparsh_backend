import mimetypes
import os
import shutil
import tempfile
import time
import base64
import io

from flask import Flask, send_file
from flask_cors import CORS
from flask_restx import Resource, Api, reqparse
from waitress import serve
from PIL import Image

from src.driver import driver_crop_pred as crop
from src.enssemble_and_compare.compare import load_models
from src.enssemble_and_compare.enssemble import predict
from src.pre_process.preprocess import driver_preprocess_pred
from src.prediction.draw_contours_on_image import draw_pred
from src.utils.utils import contour_image_folder


def create_app():
    '''
    Creates flask app as API endpoint
    '''
    app = Flask(__name__, instance_relative_config=True)

    api = Api(
        app,
        version='1.0.0',
        title='Dermatology predictor app',
        description='Dermatology predictor app',
        default='Dermatology predictor app',
        default_label=''
    )

    CORS(app)

    dermo_zip_files = reqparse.RequestParser()

    dermo_zip_files.add_argument('Image',
                                 type=str,
                                 help='Image file path',
                                 required=True)

    dermo_zip_files.add_argument('json',
                                 type=str,
                                 help='Json file path',
                                 required=True)

    @api.route('/predict')
    @api.expect(dermo_zip_files)
    class dermopredict(Resource):
        @api.expect(dermo_zip_files)
        def post(self):
            '''
            Handles post request to predict the disease
            '''
            try:
                args = dermo_zip_files.parse_args()
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
            try:
                start_time = time.time()
                file_from_request = args['Image']
                json_file_from_request = args['json']

                work_dir = tempfile.mkdtemp()
                dir = tempfile.mkdtemp()

                imageStream = io.BytesIO(base64.b64decode(file_from_request))
                imageFile = Image.open(imageStream)
                imageFile.save(os.path.join(work_dir, "image.png"))
                
                f = open(os.path.join(work_dir, "cords.json"), "wb")
                f.write(base64.b64decode(json_file_from_request))
                f.close()
                
                exp_list = crop(work_dir, dir)
                
                # dir path is where we store croped images

                pre_path = driver_preprocess_pred(dir)
                models = load_models()
                preds, prob = predict(models, pre_path, False)
                ret = ''
                for j, k in zip(preds, prob):
                    ret = ret + str(j) + ';' + str(k) + '_'

                img_pred_path = draw_pred(os.path.join(work_dir, "image.png"), exp_list, preds, prob)
                if os.path.exists(contour_image_folder):
                    shutil.rmtree(contour_image_folder)
                os.makedirs(contour_image_folder)
                shutil.copy(img_pred_path, contour_image_folder)

                prediction_image_path = os.path.join(contour_image_folder, os.listdir(contour_image_folder)[0])

                shutil.rmtree(work_dir)
                shutil.rmtree(dir)
                # mime = mimetypes.guess_type(prediction_image_path)

                return {"data": str(base64.b64encode(open(prediction_image_path, "rb").read()))[2:-1]}, 200

                # return send_file(prediction_image_path,
                #                  mimetype=mime[0],
                #                  attachment_filename=ret + os.path.basename(prediction_image_path),
                #                  as_attachment=True)

            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404

    return app


if __name__ == "__main__":
    '''
    Main function to run the app
    '''
    serve(create_app(), host='0.0.0.0', port=7777)
