package com.zzk;
import java.awt.*;
import java.net.URL;
import javax.swing.*;
public class RotateImageFrame extends JFrame {
    private Image img = null;
    private RotatePanel rotatePanel = null;
    public RotateImageFrame() {
        URL imgUrl = RotateImageFrame.class.getResource("/img/image.jpg");// 获取图片资源的路径
        img = Toolkit.getDefaultToolkit().getImage(imgUrl);   // 获取图片资源
        rotatePanel = new RotatePanel();  // 创建旋转图像的面板对象
        this.setBounds(150, 120, 380, 310);                 // 设置窗体大小和位置
        add(rotatePanel);// 在窗体上放置图像面板
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);  // 设置窗体关闭模式
        this.setTitle("旋转图像");                     // 设置窗体标题
    }
    public static void main(String[] args) {
        new RotateImageFrame().setVisible(true);
    }
    class RotatePanel extends JPanel {
        public void paint(Graphics g) {
            Graphics2D g2 = (Graphics2D) g;         // 获得Graphics2D对象
            g2.drawImage(img, 80, 10, 260, 150, this);      // 绘制指定大小的图片
            g2.rotate(Math.toRadians(10));                 // 将图片旋转10度
            g2.drawImage(img, 80, 10, 260, 150, this);      // 绘制指定大小的图片
            g2.rotate(Math.toRadians(10));                // 将图片旋转10度
            g2.drawImage(img, 80, 10, 260, 150, this);      // 绘制指定大小的图片
        }
    }
}
