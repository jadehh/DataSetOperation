#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/15 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/15  下午2:05 modify by jade
import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.84


_RESIZE_SIDE_MIN = 256
_RESIZE_SIZE_MAX = 512

def _crop(image,offset_height,offset_width,crop_height,crop_width):
    """
    裁剪图片
    :param image:
    :param offset_height:
    :param offset_width:
    :param crop_height:
    :param crop_width:
    :return:
    """
    orignal_shape = tf.shape(image)
    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image),3),
        ["图片的rank 必须等于 3"]
    )
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height,crop_width,orignal_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(orignal_shape[0],crop_height),
            tf.greater_equal(orignal_shape[1],crop_width),
        ),
        ["被裁剪的图片宽高必须大于裁剪的宽和高"]
    )
    offsets = tf.cast(tf.stack([offset_height,offset_width,0]),dtype=tf.int32)
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image,offsets,cropped_shape)
    return tf.reshape(image,cropped_shape)

def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def _smallest_size_at_least(height,width,smallest_side):
    """
    计算最小边等于smallest_side 的 新形状
    :param height: 当前图像的高
    :param width: 当前图像的宽
    :param smallest_side: python 的整数 或者 tensor ,调整大小后的最小边
    :return: 新的高和宽
    """

    smallest_side = tf.convert_to_tensor(smallest_side,dtype=tf.int32)
    height = tf.cast(height,tf.float32)
    width = tf.cast(width,tf.float32)
    smallest_side = tf.cast(smallest_side,tf.float32)
    scale = tf.cond(tf.greater(height,width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)  #smallest_side / 最小的边

    new_height = tf.cast(height * scale,tf.int32)
    new_width = tf.cast(width * scale,tf.int32)

    return new_height,new_width

def _aspect_preserving_resize(image,smallest_side):
    """
    调整图像大小，保持原本的图像纵横比
    :param image: image tensor
    :param smallest_side: python 的整数 或者 tensor ,调整大小后的最小边
    :return: 处理后的图像
    """
    smallest_side = tf.convert_to_tensor(smallest_side,dtype=tf.int32)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    new_height,new_width = _smallest_size_at_least(height,width,smallest_side)
    image = tf.expand_dims(image,0)
    resized_image = tf.image.resize(image,[new_height,new_width])
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None,None,3])
    return resized_image


def _mean_image_subtraction(image,means):
    """
    从每个图像通道中减去给定的平均值
    :param image: 图像
    :param means: 均值
    :return: 减去均值后的图像
    """
    if image.get_shape().ndims !=3 :
        raise ValueError("图像的通道数必须为3")
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError("均值列表必须等于通道数")

    channels = tf.split(axis=2,num_or_size_splits=num_channels,value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2,values=channels)

def _random_crop(image_list,crop_height,crop_width):
    """
    裁剪给定的图片列表
    :param image_list: 图片列表
    :param crop_height: 生成图片的高
    :param crop_width: 生成图片的宽
    :return: 裁剪后的图片
    """
    if not image_list:
        raise ValueError("图片列表为空")

    rank_arrsertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank,3),
            ["Wrong rank for tensor %d [expected] [actual]",i,3,image_rank])
        rank_arrsertions.append(rank_assert)

    with tf.control_dependencies([rank_arrsertions[0]]):
        image_shape = tf.shape(image_list[0])

    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height,crop_height),
            tf.greater_equal(image_width,crop_width)),
            ["Crop size greather than the image size"]
    )

    asserts = [rank_arrsertions[0],crop_size_assert]
    for i in range(1,len(image_list)):
        image = image_list[i]
        asserts.append(rank_arrsertions[i])
        with tf.control_dependencies([rank_arrsertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height,image_height),
            ["Wrong height for tensor %s [expected] [actual]",
             image.name,height,image_height]
        )
        width_assert = tf.Assert(
            tf.equal(height,image_height),
            ["Wrong width for tensor %s [expected] [actual]",
             image.name, width, image_width]
        )

        asserts.extend([height_assert,width_assert])

    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height-crop_height + 1,[])
    with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])

    offset_height = tf.random.uniform([],
                                      maxval=max_offset_height,
                                      dtype=tf.int32)

    offset_width = tf.random.uniform([],
                                     maxval=max_offset_width,
                                     dtype=tf.int32)

    return [_crop(image,offset_height,offset_width,crop_height,crop_height) for image in image_list]


def process_for_train(image,
                      output_height,
                      output_width,
                      is_train=True,
                      resize_side_min = _RESIZE_SIDE_MIN,
                      resize_side_max = _RESIZE_SIZE_MAX):
    """
    训练数据图像预处理
    :param image:
    :param output_height: 输出图像的高
    :param output_width: 输出图像的宽
    :param resize_side_min: 预处理的resize最小的值
    :param resize_side_max: 预处理的resize最大的值
    :return: 预处理后的图像
    """
    resize_side = tf.random.uniform([],
                                    minval=resize_side_min,
                                    maxval=resize_side_max+1,
                                    dtype=tf.int32)
    image = _aspect_preserving_resize(image,resize_side)
    image = _random_crop([image],output_height,output_width)[0]
    image.set_shape([output_height,output_width,3])
    image = tf.cast(image,tf.float32)
    image = tf.image.random_flip_left_right(image)
    return _mean_image_subtraction(image,[_R_MEAN,_G_MEAN,_B_MEAN])

def process_for_eval(image,output_height,output_width,resize_side_min=_RESIZE_SIDE_MIN):
    image = _aspect_preserving_resize(image, resize_side_min)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.cast(image,tf.float32)
    # return image
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


if __name__ == '__main__':
    import cv2
    image1 = cv2.imread("/home/jade/Data/VOCdevkit/Classify/aeroplane/2007_000032_0.jpg")
    cv2.imshow("result1",image1)
    for i in range(10):
        image = tf.convert_to_tensor(image1)
        image = process_for_train(image,224,224)
        cv2.imshow("result_"+str(i),image.numpy().astype("uint8"))
    cv2.waitKey(0)
    print(image.numpy())