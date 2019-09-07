package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;

import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.Image;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfWriter;

public class SetPictureBackground {
    
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\将图片设置为背景.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont bfChinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font FontChinese = new Font(bfChinese, 50, Font.BOLD,
                    BaseColor.BLUE);// 实例化字体类与设置字体大小属性
            Paragraph p = new Paragraph("下面是背景图片", FontChinese);// 创建段落对象
            p.setSpacingBefore(60); // 设置段落上边距
            p.setAlignment(com.itextpdf.text.Element.ALIGN_CENTER);
            Image image = Image.getInstance("image/gb.jpg");// 定义图片对象
            image.setAlignment(Image.UNDERLYING);// 将图片设置为背景
            document.add(image);// 向文档添加如片
            document.add(p);// 添加段落
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
