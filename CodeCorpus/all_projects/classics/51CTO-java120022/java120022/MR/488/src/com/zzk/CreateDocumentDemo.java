package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class CreateDocumentDemo {
    
    /**
     * @param args
     */
	public static void main(String[] args) {
        try {
            Document document = new Document();				// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream("c:\\创建第一个PDF文档.pdf"));// 关联文档对象与输出流
            document.open();							// 打开文档
            document.add(new Paragraph("First Document."));		// 向文档中添加内容
            document.add(new Paragraph("Success."));			// 向文档中添加内容
            document.close();							// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
        
    }

}
