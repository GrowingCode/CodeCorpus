package com.zzk;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileNameExtensionFilter;

public class BrowerPictureFrame extends JFrame {
    private static final long serialVersionUID = 2568309165342527300L;
    private JTextField tf_path;
    private BufferedImage img=null;
    private File imgFile = null;
    private DrawImagePanel imgPanel = null;
    private String filePath = null;
    private String currentFileName = null;
    private int currentFileIndex = 0;
    private List<String> fileNameList = new ArrayList<String>();
    public static void main(String args[]) {
        EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    BrowerPictureFrame frame = new BrowerPictureFrame();
                    frame.setVisible(true);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }
    
    /**
     * Create the frame
     */
    public BrowerPictureFrame() {
        super();
        setTitle("图片浏览器");
        setBounds(100, 100, 457, 328);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        imgPanel = new DrawImagePanel();
        final JPanel panel = new JPanel();
        getContentPane().add(imgPanel, BorderLayout.CENTER);
        final FlowLayout flowLayout = new FlowLayout();
        flowLayout.setAlignment(FlowLayout.LEFT);
        panel.setLayout(flowLayout);
        getContentPane().add(panel, BorderLayout.NORTH);

        final JLabel label = new JLabel();
        label.setText("选择文件：");
        panel.add(label);

        tf_path = new JTextField();
        tf_path.setPreferredSize(new Dimension(280,25));
        panel.add(tf_path);

        final JButton button = new JButton();
        button.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                JFileChooser fileChooser = new JFileChooser();// 创建文件选择器
                FileFilter filter = new FileNameExtensionFilter(
                        "图像文件（JPG/GIF/BMP)", "JPG", "JPEG", "GIF", "BMP");// 创建过滤器
                fileChooser.setFileFilter(filter);// 设置过滤器
                int flag = fileChooser.showOpenDialog(null);// 显示打开对话框
                if (flag == JFileChooser.APPROVE_OPTION) {
                    imgFile = fileChooser.getSelectedFile(); // 获取选中图片的File对象
                }
                if (imgFile != null) {
                    try {
                        img = ImageIO.read(imgFile);// 构造BufferedImage对象
                        filePath = imgFile.getParent();// 获得文件的路径
                        tf_path.setText(filePath);// 在文本框中显示路径
                        currentFileName = imgFile.getName();// 获得选择的文件名，赋值给存储当前文件的变量
                        final String extName = currentFileName.substring(currentFileName.lastIndexOf("."));// 获得文件的扩展名
                        File pathFile = new File(filePath);// 创建路径的File对象
                        String[] fileNames = pathFile.list(new FilenameFilter(){// 获得满足过滤条件的文件名数组
                            @Override
                            public boolean accept(File dir, String name) {
                                return name.endsWith(extName);// 返回满足扩展名的文件名
                            }
                        });
                        for (int i=0;i<fileNames.length;i++){// 遍历数组中的文件名
                            if (fileNames[i].equals(currentFileName)){// 判断是否为当前文件
                                currentFileIndex = i;// 记忆当前文件的索引值
                            }
                            fileNameList.add(fileNames[i]);// 将文件添加到集合列表中
                        }
                    } catch (IOException ex) {
                        ex.printStackTrace();
                    }
                }
                imgPanel.repaint();// 调用paint()方法
            }
        });
        button.setText("选    择");
        panel.add(button);

        final JPanel panel_1 = new JPanel();
        getContentPane().add(panel_1, BorderLayout.SOUTH);

        final JButton button_1 = new JButton();
        button_1.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                currentFileIndex = 0;// 第一张图片的索引
                imgFile = new File(filePath+"/"+fileNameList.get(currentFileIndex).toString());// 创建当前索引值对应图片的File对象
                try {
                    img = ImageIO.read(imgFile);// 创建当前图片的图像对象
                    imgPanel.repaint();// 调用paint()方法
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
        });
        button_1.setText("第一张");
        panel_1.add(button_1);

        final JButton button_2 = new JButton();
        button_2.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                currentFileIndex--;// 调整当前图片的索引值
                if (currentFileIndex < 0) {
                    currentFileIndex = fileNameList.size() - 1;// 当前图片索引为最后一张图片的索引
                }
                imgFile = new File(filePath+"/"+fileNameList.get(currentFileIndex).toString());// 创建当前索引值对应图片的File对象
                try {
                    img = ImageIO.read(imgFile);// 创建当前图片的图像对象
                    imgPanel.repaint();// 调用paint()方法，显示图片
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
        });
        button_2.setText("上一张");
        panel_1.add(button_2);

        final JButton button_3 = new JButton();
        button_3.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                currentFileIndex++;// 调整当前图片的索引值
                if (currentFileIndex > fileNameList.size() - 1) {
                    currentFileIndex = 0;// 当前图片索引为第一张图片的索引
                }
                imgFile = new File(filePath+"/"+fileNameList.get(currentFileIndex).toString());// 创建当前索引值对应图片的File对象
                try {
                    img = ImageIO.read(imgFile);// 创建当前图片的图像对象
                    imgPanel.repaint();// 调用paint()方法
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
        });
        button_3.setText("下一张");
        panel_1.add(button_3);

        final JButton button_4 = new JButton();
        button_4.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                currentFileIndex = fileNameList.size() - 1;// 最后一张图片的索引
                imgFile = new File(filePath+"/"+fileNameList.get(currentFileIndex).toString());// 创建当前索引值对应图片的File对象
                try {
                    img = ImageIO.read(imgFile);// 创建当前图片的图像对象
                    imgPanel.repaint();// 调用paint()方法
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
        });
        button_4.setText("最后一张");
        panel_1.add(button_4);
    }
    // 创建面板类
    class DrawImagePanel extends JPanel {
        public void paint(Graphics g) {
            if (img == null){
                return;
            }
            int imgW = img.getWidth(this);
            int imgH = img.getHeight(this);
            int panelW = getWidth();
            int panelH = getHeight();
            g.clearRect(0, 0, panelW, panelH);
            g.drawImage(img, 0, 0,imgW,imgH,this); // 绘制指定的图片
        }
    }
}
