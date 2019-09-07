package com.zzk;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.util.Random;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class ArtDesignFrame extends JFrame {
    ArtDesignPanel artDesignPanel = new ArtDesignPanel(); // 创建面板类的实例
    
    public static void main(String args[]) { // 主方法
        ArtDesignFrame frame = new ArtDesignFrame(); // 创建窗体类的实例
        frame.setVisible(true); // 显示窗体
    }
    
    public ArtDesignFrame() {
        super(); // 调用超类的构造方法
        setTitle("绘制艺术图案"); // 窗体标题
        setBounds(100, 100, 338, 230); // 窗体的显示位置和大小
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 窗体关闭方式
        add(artDesignPanel); // 将面板类的实例添加到窗体容器中
    }
    
    class ArtDesignPanel extends JPanel { // 创建内部面板类
        public void paint(Graphics g) {     // 重写paint()方法
            Graphics2D g2 = (Graphics2D)g; // 获得Graphics2D对象
            Ellipse2D.Float ellipse = new Ellipse2D.Float(-80, 5, 160, 10);// 创建椭圆对象
            Random random = new Random();// 创建随机数对象
            g2.translate(160, 90);// 平移坐标轴
            int R = random.nextInt(256);//随机产生颜色的R值
            int G = random.nextInt(256);//随机产生颜色的G值
            int B = random.nextInt(256);//随机产生颜色的B值
            Color color = new Color(R,G,B);//创建颜色对象
            g2.setColor(color);//指定颜色
            g2.draw(ellipse);// 绘制椭圆
            int i=0;
            while (i<100){
                R = random.nextInt(256);//随机产生颜色的R值
                G = random.nextInt(256);//随机产生颜色的G值
                B = random.nextInt(256);//随机产生颜色的B值
                color = new Color(R,G,B);//创建新的颜色对象
                g2.setColor(color);//指定颜色
                g2.rotate(10);// 旋转画布
                g2.draw(ellipse);// 绘制椭圆
                i++;
            }
        }
    }
}
