package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Image;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class AddPictureInTable {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\在表格中添加图片.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            Image image = Image.getInstance("image/picture.jpg");// 创建图像对象
            PdfPTable table = new PdfPTable(3);// 定义表格
            table.addCell("Text");// 添加单元格内容
            table.addCell("Picture");// 添加单元格内容
            table.addCell("Text");// 添加单元格内容
            table.addCell("This is a cell.");// 添加单元格内容
            table.addCell(image);// 向单元格中添加图像对象
            table.addCell("This is a cell.");// 添加单元格内容
            document.add(table);// 将表格添加到文档
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
