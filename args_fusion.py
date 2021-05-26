
class args():

	# 
	mult_device = False

	epochs = 70  # "number of training epochs, default is 70"
	if mult_device:
		batch_size = 16  # "batch size for training, default is 16"
		batch_size_eval = 1
		batch_size_test = 1
		nrows = 16
	else:
		batch_size = 6
		nrows = 8
		batch_size_eval = 1
		batch_size_test = 1

	train_ir_list = r"F:\Doctor_File\Image_fusion\Dense_Fuse\Fusion_image\image_blur_dataset\image_txt\imagelist_1.txt"
	train_vis_list = r"F:\Doctor_File\Image_fusion\Dense_Fuse\Fusion_image\image_blur_dataset\image_txt\imagelist_2.txt"
	train_gt_list = r"F:\Doctor_File\Image_fusion\Dense_Fuse\Fusion_image\image_blur_dataset\image_txt\imagelist_gt.txt"

	model_name = "test"
	train_ir_dir = r"F:\Doctor_File\Image_fusion\Dense_Fuse\Fusion_image\image_blur_dataset\train\image_1_gas_nis_gam2"
	train_vis_dir = r"F:\Doctor_File\Image_fusion\Dense_Fuse\Fusion_image\image_blur_dataset\train\image_2_gas_nis_gam2"
	train_gt_dir = r"F:\Doctor_File\Image_fusion\Dense_Fuse\Fusion_image\image_blur_dataset\train\GT"

	eval_ir_dir = r"F:\Doctor_File\Image_fusion\Dense_Fuse\Fusion_image\image_blur_dataset\valid\IR_1"
	eval_vis_dir = r"F:\Doctor_File\Image_fusion\Dense_Fuse\Fusion_image\image_blur_dataset\valid/VIS_1"

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





