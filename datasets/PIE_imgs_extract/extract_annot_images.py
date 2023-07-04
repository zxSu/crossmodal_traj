from datasets.PIE_imgs_extract.pie_data import PIE

pie_path = '/home/suzx/new_disks/eclipse_ws2/PIEpredict/PIE_dataset'
imdb = PIE(data_path=pie_path)
imdb.extract_and_save_images(extract_frame_type='annotated')
