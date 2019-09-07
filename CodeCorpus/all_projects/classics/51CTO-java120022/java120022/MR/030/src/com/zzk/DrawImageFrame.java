package com.zzk;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.Toolkit;
import java.net.URL;
import javax.swing.JFrame;
import javax.swing.JPanel;
public class DrawImageFrame extends JFrame {
    private Image img = null;  // 声明图像对象
    private DrawImagePanel imagePanel = null;  // 声明图像面板对象
    public static void main(String args[]) {
        DrawImageFrame frame = new DrawImageFrame();
        frame.setVisible(true);
    }
    public DrawImageFrame() {
        super();
        URL imgUrl = DrawImageFrame.class.getResource("/img/image.jpg");// 获取图片资源的路径
        img = Toolkit.getDefaultToolkit().getImage(imgUrl); // 获取图像资源
        imagePanel = new DrawImagePanel();  // 创建图像面板对象
        this.setBounds(200, 160, 316, 237); // 设置窗体大小和位置
        this.add(imagePanel); // 在窗体上添加图像面板对象
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 设置窗体关闭模式
        this.setTitle("绘制图像"); // 设置窗体标题
    }
    // 创建面板类
    class DrawImagePanel extends JPanel {
        public void paint(Graphics g) {
            g.drawImage(img, 0, 0, this); // 绘制指定的图片
        }
    }
}
