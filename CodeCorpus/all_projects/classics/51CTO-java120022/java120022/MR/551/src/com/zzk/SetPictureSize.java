package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Image;
import com.itextpdf.text.pdf.PdfWriter;

public class SetPictureSize {
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置图片大小.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            Image image = Image.getInstance("image/picture.jpg");// 创建图像对象
            image.setAlignment(Image.MIDDLE);// 居中显示图片
            image.scaleAbsolute(180, 120);// 设置图片新的宽度和高度
            document.add(image);// 向文档添加图片
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
