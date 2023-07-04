from datasets.JAAD_origin import JAAD

jaad_path = '/home/suzx/new_disks/eclipse_ws2/JAAD/JAAD-JAAD_2.0'
imdb = JAAD(data_path=jaad_path)
imdb.extract_and_save_images()
