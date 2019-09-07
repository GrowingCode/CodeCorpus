package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class FullScreen {
    
    public static void main(String args[]) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\全屏显示.pdf"));// 关联文档对象与输出流
            writer.setViewerPreferences(PdfWriter.PageModeFullScreen);// 设置阅读器以全屏模式显示
            document.open();// 打开文档
            document.add(new Paragraph("PageModelFullScreen."));// 向文档中添加内容
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
}
