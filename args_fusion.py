
class args():

	mult_device = False

	epochs = 60  # "number of training epochs, default is 70"
	if mult_device:
		batch_size = 8  # "batch size for training, default is 16"
		batch_size_eval = 1
		batch_size_test = 1
		nrows = 16
	else:
		batch_size = 4
		nrows = 8
		batch_size_eval = 1
		batch_size_test = 1

	train_ir_list = r"./image\Flickr2K\Flickr2K\train_txt\imagelist_1.txt"
	train_vis_list = r"./image\Flickr2K\Flickr2K\train_txt\imagelist_2.txt"
	train_gt_list = r"./image\Flickr2K\Flickr2K\train_txt\imagelist_gt.txt"

	model_name = "MIFFuse"
	train_ir_dir = r"./image/Flickr2K/Flickr2K/input1"
	train_vis_dir = r"./image/Flickr2K/Flickr2K/input2"
	train_gt_dir = r"./image/Flickr2K/Flickr2K/GT"

	eval_ir_dir = r"./image/TNO/TNO_IR"
	eval_vis_dir = r"./image/TNO/TNO_VI"

	save_model_dir = "models"
	save_loss_dir = "models/loss"

	lr = 1e-4  # "learning rate, default is 1e-4"
	log_interval = 10

	# for test
	model_path = r"./models/MIFFuse_parm_plk"

	test_ir_dir = r"./image/TNO/TNO_IR"
	test_vis_dir = r"./image/TNO/TNO_VI"

	# test_ir_dir = r"./image/flir/flir_ir"
	# test_vis_dir = r"./image/flir/Y_channel"

	# test_ir_dir = r"./image/CVC_14/IR"
	# test_vis_dir = r"./image/CVC_14





