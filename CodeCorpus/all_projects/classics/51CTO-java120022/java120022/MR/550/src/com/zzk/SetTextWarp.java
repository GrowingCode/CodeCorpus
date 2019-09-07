package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Image;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class SetTextWarp {
    
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置文字环绕.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            Image image = Image.getInstance("image/picture.jpg");// 创建图像对象
            image.scalePercent(33);// 设置原图像的比例
            image.setAlignment(Image.TEXTWRAP);// 将图片设置为文字环绕
            document.add(image);// 向文档添加图片
            StringBuffer sb = new StringBuffer();// 创建字符串缓存
            for (int i = 1; i <= 200; i++) {
                sb.append(i + " ");// 向字符串缓存中添加内容
            }
            Paragraph p = new Paragraph(sb.toString());// 创建段落对象
            document.add(p);// 将段落添加到文档中
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
