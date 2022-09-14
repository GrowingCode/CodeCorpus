package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.util.Random;

public class DrawPasswordWithImageApplet extends Applet {
    int WIDTH = 120;// 宽度
    int HEIGHT = 35;// 高度
    private String num = "";// 验证码
    Random random = new Random();// 实例化Random
    public void paint(Graphics g) {
        BufferedImage image = new BufferedImage(WIDTH, HEIGHT,
                BufferedImage.TYPE_INT_RGB);// 实例化BufferedImage
        Graphics gs = image.getGraphics(); // 获取Graphics类的对象
        if (!num.isEmpty()) {
            num = "";// 清空验证码
        }
        Font font = new Font("黑体", Font.BOLD, 20); // 通过Font构造字体
        gs.setFont(font);// 设置字体
        gs.fillRect(0, 0, WIDTH, HEIGHT);// 填充一个矩形
        Image img = null;// 声明图像对象
        img = getImage(getCodeBase(), "com/zzk/PPD.jpg"); // 创建图像对象
        gs.drawImage(img, 0, 0, WIDTH, HEIGHT, this);// 在缓冲图像对象上绘制图像
        // 输出随机的验证字符
        for (int i = 0; i < 4; i++) {
            char ctmp = (char) (random.nextInt(26) + 65); // 生成A~Z的字母
            num += ctmp;// 更新验证码
            Color color = new Color(20 + random.nextInt(120), 20 + random
                    .nextInt(120), 20 + random.nextInt(120));// 生成随机颜色
            gs.setColor(color); // 设置颜色
            Graphics2D gs2d = (Graphics2D) gs;// 将文字旋转指定角度
            AffineTransform trans = new AffineTransform();// 实例化AffineTransform
            trans.rotate(random.nextInt(45) * 3.14 / 180, 22 * i + 8, 7);
            float scaleSize = random.nextFloat() + 0.8f;// 缩放文字
            if (scaleSize > 1f)
                scaleSize = 1f;// 如果scaleSize大于1,则等于1
            trans.scale(scaleSize, scaleSize); // 进行缩放
            gs2d.setTransform(trans);// 设置AffineTransform对象
            gs2d.drawString(String.valueOf(ctmp), WIDTH / 6 * i + 28, HEIGHT / 2);// 画出验证码
        }
        g.drawImage(image, 85, 80, null);// 在小应用程序中绘制含有图片验证码的缓冲图像
    }
}