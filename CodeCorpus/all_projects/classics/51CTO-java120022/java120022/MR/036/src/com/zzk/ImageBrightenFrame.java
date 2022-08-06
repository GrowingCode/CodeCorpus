package com.zzk;

import java.awt.BorderLayout;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.RescaleOp;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
public class ImageBrightenFrame extends JFrame {
    private BufferedImage image;// 用于调整亮度的缓冲图像对象
    private BufferedImage oldImage;// 用于存放调整亮度之前的原缓冲图像对象
    private ImageBrightenPanel imageBrightenPanel = new ImageBrightenPanel();
    
    public static void main(String args[]) {
        ImageBrightenFrame frame = new ImageBrightenFrame();
        frame.setVisible(true);
    }
    
    public ImageBrightenFrame() {
        super();
        setBounds(100, 100, 357, 276);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setTitle("调整图片的亮度");
        Image img = null;
        try {
            img = ImageIO.read(new File("src/img/image.jpg"));  // 创建图像对象
        } catch (IOException e) {
            e.printStackTrace();
        }
        image = new BufferedImage(img.getWidth(this), img.getHeight(this),
                BufferedImage.TYPE_INT_RGB);// 创建缓冲图像对象
        image.getGraphics().drawImage(img, 0, 0, null);// 在缓冲图像对象上绘制图像
        oldImage = image;// 存储原来的图像对象，用于以后的恢复操作
        getContentPane().add(imageBrightenPanel, BorderLayout.CENTER);
        
        final JPanel panel = new JPanel();
        getContentPane().add(panel, BorderLayout.SOUTH);
        
        final JButton button = new JButton();
        button.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                float a = 1.0f;// 定义缩放因子
                float b = 5.0f;// 定义偏移量
                RescaleOp op = new RescaleOp(a,b,null);// 创建具有指定缩放因子和偏移量的 RescaleOp对象
                image = op.filter(image, null);// 对源图像中的数据进行逐像素重缩放，达到变亮的效果
                repaint();// 重新绘制图像
            }
        });
        button.setText("变    亮");
        panel.add(button);
        
        final JButton button_3 = new JButton();
        button_3.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                float a = 1.0f;// 定义缩放因子
                float b = -5.0f;// 定义偏移量
                RescaleOp op = new RescaleOp(a,b,null);// 创建具有指定缩放因子和偏移量的 RescaleOp对象
                image = op.filter(image, null);// 对源图像中的数据进行逐像素重缩放，达到变暗的效果
                repaint();// 重新绘制图像
            }
        });
        button_3.setText("变    暗");
        panel.add(button_3);
        
        final JButton button_2 = new JButton();
        button_2.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                image = oldImage;  // 获得变亮前的图像
                imageBrightenPanel.repaint();// 重新绘制原图像，即恢复为变亮前的图像
            }
        });
        button_2.setText("恢    复");
        panel.add(button_2);
        
        final JButton button_1 = new JButton();
        button_1.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                System.exit(0);
            }
        });
        button_1.setText("退    出");
        panel.add(button_1);
    }
    
    class ImageBrightenPanel extends JPanel {
        public void paint(Graphics g) {
            if (image != null) {
                g.drawImage(image, 0, 0, null);  // 将缓冲图像对象绘制到面板上
            }
        }
    }
}
